import math

import torch
import torch.nn as nn

from others.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()

        return sent_scores

class ClassifierExt(nn.Module):
    def __init__(self, hidden_size):
        super(ClassifierExt, self).__init__()
        self.linear1 = nn.Linear(hidden_size*3, hidden_size)
        self.dropout=torch.nn.Dropout(0.5)
        self.relu= nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size,1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_cls):
        h = self.linear1(x)
        h= self.relu(h)
        h= self.dropout(h)
        h = self.linear2(h).squeeze(-1)
        sent_scores = self.softmax(h) * mask_cls.float()
        return sent_scores

class ClassifierExtWithBefore(nn.Module):
    def __init__(self, hidden_size):
        super(ClassifierExtWithBefore, self).__init__()
        self.linear1 = nn.Linear(hidden_size * 3, hidden_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_cls):
        sent_scores_feature = self.linear1(x)
        # batch_size,decode_step,ext_size,hidden_size
        before_sent_feature = [torch.zeros([sent_scores_feature.size(0),sent_scores_feature.size(2),sent_scores_feature.size(3)],device=sent_scores_feature.device)]
        for i in range(1, sent_scores_feature.size(1)):
            sent_score_before = torch.max(sent_scores_feature[:, :i, :, :], dim=1)[0]
            before_sent_feature.append(sent_score_before)
        before_sent_feature = torch.stack(before_sent_feature, dim=1)
        cur_sent_feature = torch.cat([sent_scores_feature, before_sent_feature], dim=-1)
        cur_sent_feature = self.linear2(cur_sent_feature)
        cur_sent_feature = self.dropout(self.relu(cur_sent_feature))
        cur_sent_feature = self.linear3(cur_sent_feature).squeeze(-1)
        sent_scores = self.softmax(cur_sent_feature) * mask_cls.float()
        return sent_scores

    def predict(self,x, mask_cls,before_sent_feature):
        sent_scores_feature = self.linear1(x)
        # batch_size,decode_step,ext_size,hidden_size
        cur_sent_feature = torch.cat([sent_scores_feature, before_sent_feature], dim=-1)
        before_sent_feature=torch.where(before_sent_feature>sent_scores_feature,before_sent_feature,sent_scores_feature)
        cur_sent_feature = self.linear2(cur_sent_feature)
        cur_sent_feature = self.dropout(self.relu(cur_sent_feature))
        cur_sent_feature = self.linear3(cur_sent_feature).squeeze(-1)
        sent_scores = self.softmax(cur_sent_feature) * mask_cls.float()
        return sent_scores,before_sent_feature

class ClassifierExtWithBeforeScore(nn.Module):
    def __init__(self, hidden_size):
        super(ClassifierExtWithBeforeScore, self).__init__()
        self.linear1 = nn.Linear(hidden_size*3, hidden_size)
        self.dropout=torch.nn.Dropout(0.5)
        self.relu= nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size,1)
        self.linear3 = nn.Linear(3,128)
        self.linear4 = nn.Linear(3,1)
        self.softmax = nn.Softmax(dim=-1)

    def forward_old(self, x, mask_cls):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h).squeeze(-1)
        sent_scores = self.softmax(h) * mask_cls.float()
        # batch_size,decode_step,ext_size
        max_score=torch.zeros([sent_scores.size(0),sent_scores.size(-1)],dtype=torch.float).to(x.device).unsqueeze(-1)
        # print(sent_scores.size())
        # print(max_score.size())
        index = torch.arange(0,sent_scores.size(-1),dtype=torch.float).unsqueeze(0).expand(sent_scores.size(0),-1).to(x.device).unsqueeze(-1)
        final_score=[]
        for i in range(0, sent_scores.size(1)):
            score_step=sent_scores.select(1,i).unsqueeze(-1)
            # print(index.size(),score_step.size(),max_score.size())
            score_step_concat=torch.cat([index,score_step,max_score],dim=-1)
            # print(score_step_concat.size())
            score_step=self.linear4(self.dropout(self.relu(self.linear3(score_step_concat))))
            score_step = (self.softmax(score_step.squeeze(-1)) * mask_cls.float()).unsqueeze(-1)
            # print(score_step.size())
            final_score.append(score_step)
            max_score=torch.where(max_score>score_step,max_score,score_step)
        sent_scores=torch.stack(final_score,dim=1).squeeze(-1)
        # sent_scores = self.softmax(sent_scores) * mask_cls.float()
        return sent_scores

    def forward(self, x, mask_cls):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h).squeeze(-1)
        sent_scores=h
        # sent_scores = self.softmax(h) * mask_cls.float()
        # batch_size,decode_step,ext_size
        max_score = torch.zeros([sent_scores.size(0), sent_scores.size(-1)], dtype=torch.float).to(x.device)
        # print(sent_scores.size())
        # print(max_score.size())
        index = torch.arange(0, sent_scores.size(-1), dtype=torch.float).unsqueeze(0).unsqueeze(0).expand(sent_scores.size(0),sent_scores.size(1),
                                                                                             -1).to(x.device)
        index[index > 0] = 1
        max_scores = [max_score]
        # print(sent_scores.size())
        for i in range(1, sent_scores.size(1)):
            # print(sent_scores[:, :i, :])
            # print(torch.max(sent_scores[:, :i, :], dim=1))
            sent_score_before = torch.max(sent_scores[:, :i, :], dim=1)[0]
            max_scores.append(sent_score_before)
        max_scores=torch.stack(max_scores,dim=1)
        score_step_concat = torch.cat([index.unsqueeze(-1), sent_scores.unsqueeze(-1), (1-max_scores).unsqueeze(-1)], dim=-1)
        # final_score = self.linear4(self.dropout(self.relu(self.linear3(score_step_concat)))).squeeze(-1)
        final_score = self.linear4(score_step_concat).squeeze(-1)
        sent_scores = self.softmax(final_score) * mask_cls.float()
        return sent_scores

    def predict(self,x, mask_cls,max_score):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h).squeeze(-1)
        sent_scores = h
        # sent_scores = self.softmax(h) * mask_cls.float()
        # batch_size,decode_step,ext_size
        index = torch.arange(0, sent_scores.size(-1),dtype=torch.float).unsqueeze(0).expand(sent_scores.size(0),-1).to(x.device)
        index[index>0]=1
        # print(max_score.size(),index.size(),sent_scores.size())
        score_step_concat = torch.cat([index.unsqueeze(-1), sent_scores.unsqueeze(-1), (1-max_score)],dim=-1)
        # final_score = self.linear4(self.dropout(self.relu(self.linear3(score_step_concat))))
        final_score = self.linear4(score_step_concat)
        # print(max_score.size(),sent_scores.size())
        max_score = torch.where(max_score > sent_scores.unsqueeze(-1), max_score, sent_scores.unsqueeze(-1))
        sent_scores = self.softmax(final_score.squeeze(-1)) * mask_cls.float()
        return sent_scores,max_score

    def predict_old(self,x, mask_cls,max_score):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h).squeeze(-1)
        sent_scores = self.softmax(h) * mask_cls.float()
        # batch_size,decode_step,ext_size
        index = torch.arange(0, sent_scores.size(-1),dtype=torch.float).unsqueeze(0).expand(sent_scores.size(0),-1).to(x.device)
        # print(max_score.size(),index.size(),sent_scores.size())
        score_step_concat = torch.cat([index.unsqueeze(-1), sent_scores.unsqueeze(-1), max_score],dim=-1)
        score_step = self.linear4(self.dropout(self.relu(self.linear3(score_step_concat))))
        sent_scores = self.softmax(score_step.squeeze(-1)) * mask_cls.float()
        max_score = torch.where(max_score > sent_scores.unsqueeze(-1), max_score, sent_scores.unsqueeze(-1))
        return sent_scores,max_score

class ClassifierSourceType(nn.Module):
    def __init__(self, hidden_size):
        super(ClassifierSourceType, self).__init__()
        self.linear1=nn.Linear(1,hidden_size)
        self.linear2 = nn.Linear(hidden_size*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls,source_type):
        h1 = self.linear1(source_type)
        h2 = torch.cat([x,h1],-1)
        h = self.linear1(h2).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        try:
            # print(self.pe)
            # print(step)
            x=torch.index_select(self.pe,1,step).squeeze(0).unsqueeze(1)
            # print('emb:',emb.size())
            # print(torch.index_select(self.pe,1,step).size())
            # print(self.pe[:, 1].size())
            emb = emb + x
            # print(emb.size())
        except Exception as e:
            # print(e)
            if(step):
                emb = emb + self.pe[:, step][:, None, :]
            else:
                emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context,_ = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)
            # x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

class ExtTransformerEncoderSourceType(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoderSourceType, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ls = nn.Embedding(6,d_model)
        self.wo = nn.Linear(d_model*2, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask,source_type):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)
            # x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        h1 = self.ls(source_type)
        sent_scores = self.sigmoid(self.wo(torch.cat([x,h1],-1)))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
