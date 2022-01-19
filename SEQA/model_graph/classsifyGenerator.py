import torch.nn as nn
import copy
import torch
from torch.nn.init import xavier_uniform_
import numpy as np
import torch.nn.functional as F
from model_graph.decoder import TransformerDecoder,TransformerSplitQmDecoder
from model_graph.graph_transformer import GraphTransformer
from model_graph.encoder import RelationEncoder
from model_graph.transformer import Transformer
def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.Softmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator
class classsifyGenerator(nn.Module):
    def __init__(self, args, encoder,device, checkpoint=None,vocabs=None):
        super(classsifyGenerator, self).__init__()
        self.args = args
        self.device = device
        self.bert = encoder
        self.cls_vid = vocabs['tokens']['[CLS]']
        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        self.padding_idx=tgt_embeddings.padding_idx
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        if(self.args.split_qm):
            self.decoder = TransformerSplitQmDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        else:
            self.decoder = TransformerDecoder(
                    self.args.dec_layers,
                    self.args.dec_hidden_size, heads=self.args.dec_heads,
                    d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight
        # print(1111)
        if args.copy_decoder:
            if args.copy_word and args.copy_sent:
                self.p = torch.nn.Linear(self.bert.model.config.hidden_size * 3, 3)
            else:
                self.p = torch.nn.Linear(self.bert.model.config.hidden_size * 2, 1)

        self.embed_dim = self.bert.model.config.hidden_size
        if self.args.graph_transformer:
            self.relation_encoder = RelationEncoder(vocabs['relation'], args.rel_dim, self.embed_dim, args.rnn_hidden_size,
                                                    args.rnn_num_layers, args.enc_dropout)
            self.graph_encoder = GraphTransformer(args.graph_layers, self.embed_dim, args.ff_embed_dim, args.num_heads,
                                                  args.enc_dropout)
            self.snt_encoder = Transformer(args.snt_layers, self.embed_dim, args.ff_embed_dim, args.num_heads,
                                           args.enc_dropout,
                                           with_external=True)
            # self.concept_depth = nn.Embedding(256, self.embed_dim)
            self.concept_embed_layer_norm = nn.LayerNorm(self.embed_dim)

        if self.args.classify:
            self.sent_classifier = nn.Linear(self.bert.model.config.hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            if self.args.classify:
                for module in self.sent_classifier.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
            if args.copy_decoder:
            #     for module in self.attn.modules():
            #         if isinstance(module, nn.Linear):
            #             module.weight.data.normal_(mean=0.0, std=0.02)
            #         if isinstance(module, nn.Linear) and module.bias is not None:
            #             module.bias.data.zero_()
            #
                self.p.weight.data.normal_(mean=0.0, std=0.02)
                self.p.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)


    def attention_net(self, key, query):
        attn_weights = torch.bmm(key, query).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden , n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden, 1]
        context = torch.bmm(key.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context  # context : [batch_size, n_hidden]

    def attention_decoder(self,query,key):

        pass


    def encode_step(self, inp):
        sents = inp['sents']
        sents_segments = inp['sents_segments']
        # print(sents.size())
        # sent_size,batch_size,sent_len
        bts = sents.size(0)
        sent_size = sents.size(1)
        sent_len = sents.size(2)
        sents_conpact = sents.reshape(sent_size * bts, -1)
        sents_segments = sents_segments.reshape(sent_size * bts, -1)
        mask_src = ~torch.eq(sents_conpact, self.padding_idx)
        top_vec = self.bert(sents_conpact, segs=sents_segments, mask=mask_src)
        # batch_size,sent_len,hidden_size
        word_vec = top_vec.reshape(bts,sent_size*sent_len,-1)

        sents_word = sents.reshape(bts,sent_size*sent_len)

        top_vec = top_vec[:, 0:1, :]
        top_vec = top_vec.view( bts,sent_size, -1)

        # top_vec = top_vec + self.concept_depth(inp['sent_depth'])
        # sent_size,batch_size,seq_len
        mask_sent = torch.eq(inp['sents'].sum(-1), self.padding_idx)

        top_vec = self.concept_embed_layer_norm(top_vec)
        relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
        relation = relation.index_select(0, inp['relation'].contiguous().view(-1)).contiguous().view(*inp['relation'].size(), -1)
        top_vec = top_vec.transpose(0,1)
        mask_sent = mask_sent.transpose(0,1)
        # print(top_vec.size(),relation.size(),mask_sent.size())
        sent_repr = self.graph_encoder(top_vec, relation, self_padding_mask=mask_sent)
        # probe = torch.tanh(self.probe_generator(sent_repr[:1]))
        sent_repr = sent_repr[1:]
        # sent_mask = mask_sent[1:]
        sent_repr = sent_repr.transpose(0,1)
        sent_repr = sent_repr.unsqueeze(-2).expand(bts,sent_size-1,sent_len,sent_repr.size(-1))
        sent_repr = sent_repr.reshape(bts,(sent_size-1)*sent_len,sent_repr.size(-1))

        sent_repr = word_vec[:,sent_len:]+sent_repr

        return sent_repr, sents_word[:,sent_len:]

    def avgsentpooling(self,top_vec,cls_ids,sep_ids):
        sents_vec = torch.zeros([top_vec.size(0), cls_ids.size(1),top_vec.size(-1)]).to(self.device)
        # print(cls_ids.size())
        # print(sents_vec.size())
        for i in range(top_vec.size(0)):
            for j,(start,end) in enumerate(zip(cls_ids[i],sep_ids[i])):
                if end>start:
                    sents_vec[i][j] = torch.mean(top_vec[i,start:end],dim=0)
                else:
                    sents_vec[i][j] = top_vec[i,start]
        return sents_vec

    def encode_step_split(self, inp):
        # print(inp.keys())
        bts = inp['src_tokens'].size(0)
        if self.args.use_cls:
            if self.args.multidoc:
                # batch,doc_size,doc_len
                doc_size = inp['src_tokens'].size(1)
                doc_len = inp['src_tokens'].size(2)
                src_tokens = inp['src_tokens'].view(bts*doc_size,-1)
                segments = inp['segments'].view(bts*doc_size,-1)
                mask_q = ~torch.eq(src_tokens, self.padding_idx)
                top_vec = self.bert(src_tokens, segments, mask_q)
                #batch*doc_size,doc_len,hidden_size
                top_vec = top_vec.view(bts,doc_size*doc_len,-1)
                mask_q = mask_q.view(bts, doc_size * doc_len)
                inp['src_tokens']=inp['src_tokens'].view(bts,doc_size*doc_len)
            else:
                mask_q = ~torch.eq(inp['src_tokens'], self.padding_idx)
                top_vec = self.bert(inp['src_tokens'], inp['segments'], mask_q)
            # print(inp['cls_ids'])
            # print(inp['sep_ids'])
            if self.args.avg_sent:
                sents_vec=self.avgsentpooling(top_vec,inp['cls_ids'],inp['sep_ids'])
                # print(sents_vec)
            else:
                sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1),inp['cls_ids']]
            # print(torch.tensor([self.cls_vid]).unsqueeze(1))
            # print(sents_vec.size())
            if self.args.graph_transformer:
                head_vec = self.bert(torch.tensor([self.cls_vid]).unsqueeze(1).expand(bts,1).to(self.device),None,None)

                sent_vec = torch.cat([head_vec,sents_vec],dim=1)
                # print(inp['mask_cls'])
                # print(inp['cls_ids'])
                mask_sent = torch.cat([torch.tensor([0]).unsqueeze(1).expand(bts,1).to(self.device),inp['mask_cls']],dim=-1).byte()
                # print(mask_sent)
                sent_vec = self.concept_embed_layer_norm(sent_vec)
                relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
                relation = relation.index_select(0, inp['relation'].contiguous().view(-1)).contiguous().view(
                    *inp['relation'].size(), -1)
                sent_vec = sent_vec.transpose(0, 1)
                mask_sent = mask_sent.transpose(0, 1)
                # print(top_vec.size(),relation.size(),mask_sent.size())
                # print(sent_vec.size(),relation.size(),mask_sent.size())
                sent_repr = self.graph_encoder(sent_vec, relation, self_padding_mask=mask_sent)
                # probe = torch.tanh(self.probe_generator(sent_repr[:1]))
                sent_repr = sent_repr[1:]
                # sent_mask = mask_sent[1:]
                sent_repr = sent_repr.transpose(0, 1)

                mask_sent = mask_sent[1:]
                mask_sent = mask_sent.transpose(0, 1)
            else:
                sent_repr = sents_vec
                mask_sent = inp['mask_cls'].byte()

            return top_vec, inp['src_tokens'], sent_repr, ~mask_q, mask_sent
        else:
            sents = inp['sents']
            # print(sents.size())
            # sent_size,batch_size,sent_len
            sent_size = sents.size(1)
            sent_len = sents.size(2)
            sents_conpact = sents.reshape(sent_size * bts, -1)
            mask_src = ~torch.eq(sents_conpact, self.padding_idx)
            top_vec = self.bert(sents_conpact, segs=None, mask=mask_src)
            # batch_size,sent_len,hidden_size
            word_vec = top_vec.reshape(bts,sent_size*sent_len,-1)

            sents_word = sents.reshape(bts,sent_size*sent_len)

            top_vec = top_vec[:, 0:1, :]
            top_vec = top_vec.view( bts,sent_size, -1)

            # top_vec = top_vec + self.concept_depth(inp['sent_depth'])
            # sent_size,batch_size,seq_len
            mask_sent = torch.eq(inp['sents'].sum(-1), self.padding_idx)

            top_vec = self.concept_embed_layer_norm(top_vec)
            relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
            relation = relation.index_select(0, inp['relation'].contiguous().view(-1)).contiguous().view(*inp['relation'].size(), -1)
            top_vec = top_vec.transpose(0,1)
            mask_sent = mask_sent.transpose(0,1)
            # print(top_vec.size(),relation.size(),mask_sent.size())
            sent_repr = self.graph_encoder(top_vec, relation, self_padding_mask=mask_sent)
            # probe = torch.tanh(self.probe_generator(sent_repr[:1]))
            sent_repr = sent_repr[1:]
            # sent_mask = mask_sent[1:]
            sent_repr = sent_repr.transpose(0,1)

            mask_sent = mask_sent[1:]
            mask_sent = mask_sent.transpose(0,1)
            # sent_repr = sent_repr.unsqueeze(-2).expand(bts,sent_size-1,sent_len,sent_repr.size(-1))
            # sent_repr = sent_repr.reshape(bts,(sent_size-1)*sent_len,sent_repr.size(-1))
            #
            # sent_repr = word_vec[:,sent_len:]+sent_repr
            if self.args.encode_q:
                mask_q = ~torch.eq(inp['src_tokens'], self.padding_idx)
                top_vec =self.bert(inp['src_tokens'], inp['segments'], mask_q)
                return top_vec,inp['src_tokens'],sent_repr,~mask_q,mask_sent
            else:
                mask_word = torch.eq(sents_word[:,sent_len:sent_len*2], self.padding_idx)
                return word_vec[:,sent_len:sent_len*2], sents_word[:,sent_len:sent_len*2],sent_repr,mask_word,mask_sent

    def encode_word(self, inp):
        # print(inp.keys())
        bts = inp['src_tokens'].size(0)
        if self.args.multidoc:
            # batch,doc_size,doc_len
            doc_size = inp['src_tokens'].size(1)
            doc_len = inp['src_tokens'].size(2)
            src_tokens = inp['src_tokens'].view(bts*doc_size,-1)
            segments = inp['segments'].view(bts*doc_size,-1)
            mask_q = ~torch.eq(src_tokens, self.padding_idx)
            top_vec = self.bert(src_tokens, segments, mask_q)
            #batch*doc_size,doc_len,hidden_size
            top_vec = top_vec.view(bts,doc_size*doc_len,-1)
            mask_q = mask_q.view(bts, doc_size * doc_len)
            inp['src_tokens']=inp['src_tokens'].view(bts,doc_size*doc_len)
        else:
            mask_q = ~torch.eq(inp['src_tokens'], self.padding_idx)
            top_vec = self.bert(inp['src_tokens'], inp['segments'], mask_q)
        # print(inp['cls_ids'])
        # print(inp['sep_ids'])
        return top_vec, inp['src_tokens'], ~mask_q.unsqueeze(1)

    def encode_step1(self, inp):
        # batch_size,sent_size,sent_len
        sents = inp['sents'][:,1:]
        # print(sents.size())
        # sent_size,batch_size,sent_len
        bts = sents.size(0)
        sent_size = sents.size(1)
        sent_len = sents.size(2)

        sents_conpact = sents.reshape(sent_size * bts, -1)
        mask_src = ~torch.eq(sents_conpact, self.padding_idx)
        top_vec = self.bert(sents_conpact, segs=None, mask=mask_src)
        top_vec = top_vec.reshape(bts, sent_size,sent_len, -1)
        top_vec = top_vec.reshape(bts,sent_size*sent_len,-1)

        sents = sents.reshape(bts,sent_size*sent_len)
        # mask_sent = torch.eq(sents, self.padding_idx)
        sent_repr=top_vec
        sent_repr = sent_repr
        # sent_mask = mask_sent
        return sent_repr,sents


    def forward(self, data):
        if(self.args.split_qm):
            word_vec,words,sent_repr,mask_word,mask_sent = self.encode_step_split(data)
            max_node_size = sent_repr.size(1)
            dec_state = self.decoder.init_decoder_state(words, word_vec)
            decoder_outputs, state, attn_word,attn_sent = self.decoder(data['token_in'], sent_repr, word_vec,dec_state,material_masks = mask_sent, memory_masks = mask_word)
            out = self.generator(decoder_outputs)
            if self.args.copy_decoder:
                attn_src = torch.mean(attn_sent, 1)
                attn_context = torch.matmul(attn_src, sent_repr)
                index = data['copy_seq'].unsqueeze(1).expand(-1, decoder_outputs.size(1), -1)

                if self.args.copy_word and self.args.copy_sent:
                    attn_word = torch.mean(attn_word, 1)
                    attn_context_word = torch.matmul(attn_word, word_vec)
                    attn_value_src = torch.zeros([attn_word.size(0), attn_word.size(1), self.vocab_size]).to(self.device)
                    # print(words.size(),attn_word.size(),attn_value_src.size())
                    index_word = words.unsqueeze(1).expand(-1, decoder_outputs.size(1), -1)
                    attn_value_src = attn_value_src.scatter_add(2, index_word, attn_word)

                    p = torch.softmax(self.p(torch.cat([decoder_outputs, attn_context_word, attn_context], -1)), dim=-1)
                    p_gen = p[:, :, 0].unsqueeze(-1) * out
                    p_copy_word = p[:, :, 1].unsqueeze(-1) * attn_value_src
                    probs = p_gen + p_copy_word
                    out = probs/(p[:, :, 0].unsqueeze(-1)+p[:, :, 1].unsqueeze(-1))
                    p_copy_sent = p[:, :, 2].unsqueeze(-1) * attn_src
                    ext_probs = probs.new_zeros((1, 1, max_node_size)).expand(probs.size(0), probs.size(1), -1)
                    probs = torch.cat([probs, ext_probs], -1)
                    out_cat = probs.scatter_add(2, index, p_copy_sent)
                    outs = torch.log(out_cat)
                elif self.args.copy_sent:
                    p = torch.sigmoid(self.p(torch.cat([decoder_outputs, attn_context], -1)))
                    probs = p*out
                    # print(attn_context.size())
                    ext_probs = probs.new_zeros((1, 1, max_node_size)).expand(probs.size(0), probs.size(1), -1)
                    probs = torch.cat([probs, ext_probs], -1)
                    copy_probs = (1-p)*attn_src
                    out_cat = probs.scatter_add(2, index, copy_probs)
                    outs = torch.log(out_cat)
                else:
                    attn_word = torch.mean(attn_word, 1)
                    attn_context_word = torch.matmul(attn_word, word_vec)
                    attn_value_src = torch.zeros([attn_word.size(0), attn_word.size(1), self.vocab_size]).to(
                        self.device)
                    # print(words.size(),attn_word.size(),attn_value_src.size())
                    index_word = words.unsqueeze(1).expand(-1, decoder_outputs.size(1), -1)
                    attn_value_src = attn_value_src.scatter_add(2, index_word, attn_word)
                    # print(decoder_outputs, attn_context_word)
                    p = torch.sigmoid(self.p(torch.cat([decoder_outputs, attn_context_word], -1)))
                    outs = p * out + (1 - p) * attn_value_src
                    outs = torch.log(outs)
                outs[outs == float("-inf")] = -300  # -10e20
            else:
                outs=torch.log(out)
                # print(p)
                # outs = p * out + (1 - p) * attn_value_src
        else:
            if self.args.classify:
                word_vec, words, sent_repr, mask_word, mask_sent = self.encode_step_split(data)
            else:
                word_vec, words, mask_word = self.encode_word(data)
            dec_state_src = self.decoder.init_decoder_state(words, word_vec)
            decoder_outputs, state_src, attn_dist_src = self.decoder(data['token_in'], word_vec, dec_state_src,
                                                                             memory_masks=mask_word)
            out = self.generator(decoder_outputs)
            if self.args.copy_decoder:
                if self.args.copy_word:
                    attn_word = torch.mean(attn_dist_src, 1)
                    attn_context_word = torch.matmul(attn_word, word_vec)
                    attn_value_src = torch.zeros([attn_word.size(0), attn_word.size(1), self.vocab_size]).to(
                        self.device)
                    # print(words.size(),attn_word.size(),attn_value_src.size())
                    index_word = words.unsqueeze(1).expand(-1, decoder_outputs.size(1), -1)
                    attn_value_src = attn_value_src.scatter_add(2, index_word, attn_word)
                    # print(decoder_outputs, attn_context_word)
                    p = torch.sigmoid(self.p(torch.cat([decoder_outputs, attn_context_word], -1)))
                    out = p * out + (1 - p) * attn_value_src
            outs = torch.log(out)

        if self.args.classify:
            sent_logit = self.sigmoid(self.sent_classifier(sent_repr).squeeze(-1)) * mask_sent.float()
            return decoder_outputs,outs,torch.log(out),sent_logit
        else:
            if self.args.copy_decoder and self.args.copy_sent:
                return decoder_outputs, outs, torch.log(out)
            else:
                return decoder_outputs, outs

    def predict(self,decoder_input, src_features, dec_states,step,mask_word,src):
        dec_out, dec_states, attn_word = self.decoder(decoder_input, src_features, dec_states,
                                                            step=step,memory_masks=mask_word)
        probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
        if self.args.copy_decoder:
            attn_word = torch.mean(attn_word, 1)
            attn_context_word = torch.matmul(attn_word, src_features)
            attn_value_src = torch.zeros([attn_word.size(0), self.vocab_size]).to(self.device)
            attn_value_src = attn_value_src.scatter_add(1, src, attn_word.squeeze(1))

            p = torch.sigmoid(self.p(torch.cat([dec_out.transpose(0, 1).squeeze(0), attn_context_word.squeeze(1)], -1)))
            probs =p * probs+(1-p)*attn_value_src
        return probs,dec_out,dec_states

    def updateExtToken(self,decoder_input,local_idx2tokenid,src_features, dec_states,sent_repr,mask_sent,mask_word,step):
        step=step.cpu().numpy().tolist()
        decs=[]
        for i in range(decoder_input.size(0)):
            dec = decoder_input.select(0,i).cpu().item()
            if dec in local_idx2tokenid[i]:
                decs.append(local_idx2tokenid[i][dec])
            else:
                decs.append([dec])
        max_len= max(len(x) for x in decs)
        steps=[]
        for i in range(len(decs)):
            steps.append([step[i]] * (max_len - len(decs[i])) + [step[i] + s for s in range(len(decs[i]))])
            decs[i]=[self.padding_idx]* (max_len - len(decs[i]))+decs[i]

        decs = torch.tensor(decs).to(self.device)
        steps = torch.tensor(steps).to(self.device)
        decs_pre = decs[:, :-1]
        steps_pre = steps[:, :-1]
        decs_then = decs[:, -1:]
        steps_then = steps[:, -1]
        for i in range(decs_pre.size(1)):
            decoder_input = decs_pre.select(1,i).unsqueeze(-1)
            step = steps_pre.select(1,i)
            # print(decoder_input)
            # print(step)
            dec_out, dec_states, attn_word, attn_sent = self.decoder(decoder_input, sent_repr, src_features, dec_states,
                                                                     material_masks=mask_sent, memory_masks=mask_word,
                                                                     step=step)
        return decs_then,steps_then,dec_states

    # def updateExtToken(self,decoder_input,local_idx2tokenid,src_features, dec_states,sent_repr,mask_sent,mask_word,step):
    #     step=step.cpu().numpy().tolist()
    #     decs=[]
    #     for i in range(decoder_input.size(0)):
    #         dec = decoder_input.select(0,i).cpu().item()
    #         if dec in local_idx2tokenid[i]:
    #             decs.append([local_idx2tokenid[i][dec][0]])
    #         else:
    #             decs.append([dec])
    #     # max_len= max(len(x) for x in decs)
    #     # steps=[]
    #     # for i in range(len(decs)):
    #     #     steps.append([step[i]] * (max_len - len(decs[i])) + [step[i] + s for s in range(len(decs[i]))])
    #     #     decs[i]=[self.padding_idx]* (max_len - len(decs[i]))+decs[i]
    #
    #     decs = torch.tensor(decs).to(self.device)
    #     step = torch.tensor(step).to(self.device)
    #     # decs_pre = decs[:, :-1]
    #     # steps_pre = steps[:, :-1]
    #     # decs_then = decs[:, -1:]
    #     # steps_then = steps[:, -1]
    #     # for i in range(decs_pre.size(1)):
    #     #     decoder_input = decs_pre.select(1,i).unsqueeze(-1)
    #     #     step = steps_pre.select(1,i)
    #     #     # print(decoder_input)
    #     #     # print(step)
    #     #     dec_out, dec_states, attn_word, attn_sent = self.decoder(decoder_input, sent_repr, src_features, dec_states,
    #     #                                                              material_masks=mask_sent, memory_masks=mask_word,
    #     #                                                              step=step)
    #     # print(decs)
    #     # print(step)
    #     return decs,step,dec_states

    def updateExtTokenPre(self,decoder_input,local_idx2tokenid,copy_steps):
        decs = []
        for i in range(decoder_input.size(0)):
            step = copy_steps[i]
            dec = decoder_input.select(0, i).cpu().item()
            if dec in local_idx2tokenid[i]:
                # print(step)
                # print(local_idx2tokenid[i][dec])
                decs.append([local_idx2tokenid[i][dec][step]])
                copy_steps[i]=copy_steps[i]+1
                if copy_steps[i]>=len(local_idx2tokenid[i][dec]):
                    copy_steps[i]=0
            else:
                decs.append([dec])
        decs = torch.tensor(decs).to(self.device)
        return decs

    def updateExtTokenAfter(self,probs,copy_steps,decoder_input,pre_prob):
        copy_steps = torch.tensor(copy_steps).to(self.device).unsqueeze(-1).expand(probs.size(0),probs.size(1))
        # batch
        # batch,vocab_size
        new_pre_prob = torch.zeros_like(probs)
        # print(probs.topk(probs.size(0), dim=-1))
        index = decoder_input
        pre_prob=pre_prob.gather(1, index)
        # pre_prob=pre_prob*1.1
        # print(pre_prob)
        # pre_prob = probs.max(-1)[0].unsqueeze(-1)
        # print(pre_prob)
        new_pre_prob.scatter_(1, index, pre_prob)
        # new_pre_prob.scatter_(1, index, torch.full_like(index.float(),0.7))
        # new_pre_prob.scatter_(1, index, torch.ones_like(index).float())
        # pre_prob = torch.zeros_like(probs)
        # index = decoder_input
        # pre_prob = pre_prob.scatter_add(1, index, 1)

        probs = torch.where(copy_steps == 0,probs,new_pre_prob)
        return probs


    def predict_split_qm(self,decoder_input, src_features, dec_states,sent_repr,mask_sent,mask_word,step,copy_seq=None,local_idx2tokenid=None,copy_steps=None,pre_prob=None,src=None):
        decoder_input_new = decoder_input
        max_node_size = sent_repr.size(1)
        if self.args.copy_decoder and self.args.copy_sent:
            tot_ext = decoder_input.max().item()
            if tot_ext >= self.vocab_size:
                decoder_input_new, step, dec_states = self.updateExtToken(decoder_input, local_idx2tokenid, src_features,
                                                                      dec_states, sent_repr, mask_sent, mask_word, step)
                # print(copy_steps)
                # decoder_input_new = self.updateExtTokenPre(decoder_input,local_idx2tokenid,copy_steps)
                # print(copy_steps)

        dec_out, dec_states, attn_word, attn_sent = self.decoder(decoder_input_new, sent_repr, src_features, dec_states,
                                                                    material_masks=mask_sent, memory_masks=mask_word,step=step)
        # print(dec_out.size(),step.size(),sent_repr.size(),src_features.size(),mask_sent.size(),mask_word.size())
        probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
        # print(decoder_input.size(),probs.size())

        if self.args.copy_decoder:
            if self.args.copy_word and self.args.copy_sent:

                attn_src = torch.mean(attn_sent, 1)
                attn_context = torch.matmul(attn_src, sent_repr)
                index = copy_seq
                attn_word = torch.mean(attn_word, 1)
                attn_context_word = torch.matmul(attn_word, src_features)
                attn_value_src = torch.zeros([attn_word.size(0), self.vocab_size]).to(self.device)
                attn_value_src = attn_value_src.scatter_add(1, src, attn_word.squeeze(1))
                p = torch.softmax(self.p(torch.cat(
                    [dec_out.transpose(0, 1).squeeze(0), attn_context_word.squeeze(1), attn_context.squeeze(1)], -1)),
                                  dim=-1)
                p_gen = p[:, 0].unsqueeze(-1) * probs
                p_copy_word = p[:, 1].unsqueeze(-1) * attn_value_src
                probs = p_gen + p_copy_word
                # probs = probs / (p[:, 0].unsqueeze(-1) + p[:, 1].unsqueeze(-1))

                p_copy_sent = p[:, 2].unsqueeze(-1) * attn_src.squeeze(1)
                ext_probs = probs.new_zeros((probs.size(0), max_node_size))
                probs = torch.cat([probs, ext_probs], -1)
                # print(probs.size(),index_sent.size(),p_copy_sent.size())
                probs = probs.scatter_add(1, index, p_copy_sent)
            elif self.args.copy_sent:
                attn_src = torch.mean(attn_sent, 1)
                attn_context = torch.matmul(attn_src, sent_repr)
                index = copy_seq
                p = torch.sigmoid(self.p(torch.cat([dec_out.transpose(0, 1).squeeze(0), attn_context.squeeze(1)], -1)))
                probs = p * probs
                ext_probs = probs.new_zeros((1, max_node_size)).expand(probs.size(0), -1)
                # print(probs.size(),ext_probs.size())
                probs = torch.cat([probs, ext_probs], -1)
                # print('attn:',attn_src)
                copy_probs = (1 - p) * attn_src.squeeze(1)
                probs = probs.scatter_add(1, index, copy_probs)
            else:
                attn_word = torch.mean(attn_word, 1)
                attn_context_word = torch.matmul(attn_word, src_features)
                attn_value_src = torch.zeros([attn_word.size(0), self.vocab_size]).to(self.device)
                attn_value_src = attn_value_src.scatter_add(1, src, attn_word.squeeze(1))

                p = torch.sigmoid(
                    self.p(torch.cat([dec_out.transpose(0, 1).squeeze(0), attn_context_word.squeeze(1)], -1)))
                probs = p * probs + (1 - p) * attn_value_src
        # if step>0 and self.args.copy_decoder:
            # a=probs.topk(probs.size(0), dim=-1)
            # probs=self.updateExtTokenAfter(probs, copy_steps, decoder_input,pre_prob)
            # b=probs.topk(probs.size(0), dim=-1)
            # print(b)

        return probs,dec_out,dec_states,step