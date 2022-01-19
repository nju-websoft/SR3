import torch.nn as nn
import copy
import torch
from torch.nn.init import xavier_uniform_
import numpy as np
import torch.nn.functional as F
from model_graph.decoder import TransformerDecoder,TransformerDecoderSent
from model_graph.graph_transformer import GraphTransformer
from model_graph.encoder import RelationEncoder
from model_graph.transformer import Transformer
from others.neural import MultiHeadedAttention
def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.Softmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class AbsWordCopySummarizer(nn.Module):
    def __init__(self, args, encoder,device, checkpoint=None):
        super(AbsWordCopySummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = encoder

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        self.padding_idx = tgt_embeddings.padding_idx
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
        self.decoder_src = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder_src.embeddings.weight
        # print(1111)
        if args.copy_decoder:
            self.p = torch.nn.Linear(self.bert.model.config.hidden_size * 2, 1)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder_src.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            if args.copy_decoder:
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
                self.decoder_src.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder_src.embeddings.weight
        self.to(device)

    def encode_word(self, inp):
        mask_q = ~torch.eq(inp['src_tokens'], self.padding_idx)
        top_vec = self.bert(inp['src_tokens'], inp['segments'], mask_q)
        # print(mask_q)
        return top_vec, inp['src_tokens'], ~mask_q

    def forward(self, data):
        word_vec,words,mask_word = self.encode_word(data)
        mask_word = mask_word.unsqueeze(1)
        dec_state_src = self.decoder_src.init_decoder_state(words, word_vec)
        decoder_outputs_src, state_src, attn_dist_src = self.decoder_src(data['token_in'], word_vec, dec_state_src,
                                                                         memory_masks=mask_word)
        out = self.generator(decoder_outputs_src)
        if self.args.copy_decoder:
            attn_word = torch.mean(attn_dist_src, 1)
            attn_context_word = torch.matmul(attn_word, word_vec)
            attn_value_src = torch.zeros([attn_word.size(0), attn_word.size(1), self.vocab_size]).to(self.device)
            # print(words.size(),attn_word.size(),attn_value_src.size())
            index_word = words.unsqueeze(1).expand(-1, decoder_outputs_src.size(1), -1)
            attn_value_src = attn_value_src.scatter_add(2, index_word, attn_word)

            p = torch.sigmoid(self.p(torch.cat([decoder_outputs_src, attn_context_word], -1)))
            out = p*out+(1-p)*attn_value_src
        # outs = torch.log(out) # -10e20
            if torch.isnan(decoder_outputs_src).any():
                print(data['token_in'])
                print(word_vec)
                print(mask_word)
        if self.args.copy_decoder:
            return decoder_outputs_src, out,attn_context_word
        else:
            return decoder_outputs_src, out,None

    def predict(self,decoder_input, src_features, dec_states,step,mask_word,src):
        dec_out, dec_states, attn_word = self.decoder_src(decoder_input, src_features, dec_states,
                                                            step=step,memory_masks=mask_word)
        probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
        if self.args.copy_decoder:
            attn_word = torch.mean(attn_word, 1)
            attn_context_word = torch.matmul(attn_word, src_features)
            attn_value_src = torch.zeros([attn_word.size(0), self.vocab_size]).to(self.device)
            attn_value_src = attn_value_src.scatter_add(1, src, attn_word.squeeze(1))

            p = torch.sigmoid(self.p(torch.cat([dec_out.transpose(0, 1).squeeze(0), attn_context_word.squeeze(1)], -1)))
            probs =p * probs+(1-p)*attn_value_src
            return probs,dec_out,dec_states,attn_context_word
        else:
            return probs, dec_out, dec_states,None

class SentCopySummarizer(nn.Module):
    def __init__(self, args, encoder,device, checkpoint=None,gencheckpoint=None,vocabs=None):
        super(SentCopySummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = encoder

        if gencheckpoint==None and checkpoint==None:
            self.tunegen = True
        else:
            self.tunegen = False
        self.wordCopySumm = AbsWordCopySummarizer(args, encoder,device, gencheckpoint)
        self.padding_idx = self.wordCopySumm.padding_idx
        self.vocab_size = self.wordCopySumm.vocab_size
        if self.args.sent_attn and  self.args.split_qm:
            self.decoder_sent=TransformerDecoderSent(
                1,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, padding_idx=self.padding_idx)
        elif self.args.split_qm:
            self.decoder_sent = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.dec_heads,
                d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=self.wordCopySumm.decoder_src.embeddings)

        if args.copy_decoder:
            self.p = torch.nn.Linear(self.bert.model.config.hidden_size * 3, 1)

        self.embed_dim = self.bert.model.config.hidden_size
        if self.args.split_qm:
            self.relation_encoder = RelationEncoder(vocabs['relation'], args.rel_dim, self.embed_dim, args.rnn_hidden_size,
                                                    args.rnn_num_layers, args.enc_dropout)
            self.graph_encoder = GraphTransformer(args.graph_layers, self.embed_dim, args.ff_embed_dim, args.num_heads,
                                                  args.enc_dropout)
            self.snt_encoder = Transformer(args.snt_layers, self.embed_dim, args.ff_embed_dim, args.num_heads,
                                           args.enc_dropout,
                                           with_external=True)
            # self.concept_depth = nn.Embedding(256, self.embed_dim)
            self.concept_embed_layer_norm = nn.LayerNorm(self.embed_dim)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            if self.args.split_qm:
                for module in self.decoder_sent.modules():
                        if isinstance(module, (nn.Linear, nn.Embedding)):
                            module.weight.data.normal_(mean=0.0, std=0.02)
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)
                        if isinstance(module, nn.Linear) and module.bias is not None:
                            module.bias.data.zero_()
            if args.copy_decoder:
                self.p.weight.data.normal_(mean=0.0, std=0.02)
                self.p.bias.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.wordCopySumm.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.wordCopySumm.bert.model.embeddings.word_embeddings.weight)
                if self.args.split_qm and not self.args.sent_attn:
                    self.decoder_sent.embeddings = tgt_embeddings
        self.to(device)


    def encode_step(self, inp):
        sents = inp['sents']
        # print(sents.size())
        # sent_size,batch_size,sent_len
        bts = sents.size(0)
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
        sent_repr = sent_repr.unsqueeze(-2).expand(bts,sent_size-1,sent_len,sent_repr.size(-1))
        sent_repr = sent_repr.reshape(bts,(sent_size-1)*sent_len,sent_repr.size(-1))

        sent_repr = word_vec[:,sent_len:]+sent_repr

        return sent_repr, sents_word[:,sent_len:]


    def encode_step_split(self, inp):
        sents = inp['sents']
        # print(sents.size())
        # sent_size,batch_size,sent_len
        bts = sents.size(0)
        sent_size = sents.size(1)
        sent_len = sents.size(2)
        sents_conpact = sents.reshape(sent_size * bts, -1)
        mask_src = ~torch.eq(sents_conpact, self.padding_idx)
        if self.tunegen:
            top_vec = self.wordCopySumm.bert(sents_conpact, segs=None, mask=mask_src)
        else:
            with torch.no_grad():
                top_vec = self.wordCopySumm.bert(sents_conpact, segs=None, mask=mask_src)
        # batch_size,sent_len,hidden_size
        top_vec = top_vec[:, 0:1, :]
        top_vec = top_vec.view( bts,sent_size, -1)
        mask_sent = torch.eq(inp['sents'].sum(-1), self.padding_idx)

        top_vec_norm = self.concept_embed_layer_norm(top_vec)
        if torch.isnan(top_vec_norm).any():
            print('top_vec_norm')
            print(top_vec)
            print(torch.isnan(top_vec).any())
            print(torch.isnan(self.concept_embed_layer_norm.weight).any())
            print(torch.isnan(self.concept_embed_layer_norm.bias).any())

        top_vec=top_vec_norm
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

        return sent_repr,mask_sent


    def forward(self, data):
        if (self.args.split_qm):
            if self.tunegen:
                if self.args.copy_decoder:
                    decoder_outputs_src, out, attn_context_word = self.wordCopySumm(data)
                else:
                    decoder_outputs_src, out = self.wordCopySumm(data)
            else:
                with torch.no_grad():
                    if self.args.copy_decoder:
                        decoder_outputs_src, out, attn_context_word = self.wordCopySumm(data)
                    else:
                        decoder_outputs_src, out = self.wordCopySumm(data)

            sent_repr, mask_sent = self.encode_step_split(data)
            dec_state_sent = self.decoder_sent.init_decoder_state(~mask_sent, sent_repr)
            mask_sent = mask_sent.unsqueeze(1)
            if self.args.sent_attn:
                decoder_outputs_sent, state_sent, attn_dist_sent = self.decoder_sent(decoder_outputs_src, sent_repr,
                                                                                     dec_state_sent,memory_masks=mask_sent)
            else:
                decoder_outputs_sent, state_sent, attn_dist_sent = self.decoder_sent(data['token_in'], sent_repr, dec_state_sent, memory_masks=mask_sent)
            if self.args.copy_decoder:
                attn_src = torch.mean(attn_dist_sent, 1)
                attn_context = torch.matmul(attn_src, sent_repr)
                index = data['copy_seq'].unsqueeze(1).expand(-1, decoder_outputs_src.size(1), -1)

                p = torch.sigmoid(self.p(torch.cat([decoder_outputs_src, attn_context_word,attn_context], -1)))
                probs = p*out
                # print(attn_context.size())
                ext_probs = probs.new_zeros((1, 1, self.args.max_node_size)).expand(probs.size(0), probs.size(1), -1)
                probs = torch.cat([probs, ext_probs], -1)
                copy_probs = (1-p)*attn_src
                out_cat = probs.scatter_add(2, index, copy_probs)

                outs = torch.log(out_cat)
                # print(torch.isnan(outs))
                if torch.isnan(outs).any():
                    print(p)
                    print(decoder_outputs_src)
                    print(attn_context_word)
                    print(attn_src)
                    print(sent_repr)
                    for name, param in self.named_parameters():
                        print(name,torch.isnan(param).any())
                assert not torch.isnan(outs).any()
                outs[outs == float("-inf")] = -300  # -10e20
                return decoder_outputs_src, outs,torch.log(out)

    def updateExtToken(self,decoder_input,local_idx2tokenid,src_features, dec_states_src,dec_states_sent,sent_repr,mask_sent,mask_word,step):
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

        # decs = []
        # for i in range(decoder_input.size(0)):
        #     dec = decoder_input.select(0, i).cpu().item()
        #     decs.append([dec])
        # max_len = 5
        # steps = []
        # for i in range(len(decs)):
        #     steps.append([step[i]] * max_len)
        #     decs[i] = [self.padding_idx] * (max_len - len(decs[i])) + decs[i]

        decs = torch.tensor(decs).to(self.device)
        steps = torch.tensor(steps).to(self.device)
        decs_pre = decs[:, :-1]
        steps_pre = steps[:, :-1]
        decs_then = decs[:, -1:]
        steps_then = steps[:, -1]
        for i in range(decs_pre.size(1)):
            decoder_input = decs_pre.select(1,i).unsqueeze(-1)
            # print(decoder_input)
            step = steps_pre.select(1,i)
            dec_out, dec_states_src, _ = self.wordCopySumm.decoder_src(decoder_input, src_features, dec_states_src,memory_masks=mask_word,step=step)
            if self.args.sent_attn:
                _, dec_states_sent, _ = self.decoder_sent(dec_out, sent_repr,
                                                                dec_states_sent, memory_masks=mask_sent,
                                                                step=step)
            else:
                dec_out, dec_states_sent, _ = self.decoder_sent(decoder_input, sent_repr,
                                                                                 dec_states_sent, memory_masks=mask_sent,
                                                                                 step=step)
        return decs_then,steps_then,dec_states_src,dec_states_sent

    def predict_split_qm(self,decoder_input, src_features, dec_states_src,dec_states_sent,sent_repr,mask_sent,mask_word,step,copy_seq=None,local_idx2tokenid=None,copy_steps=None,pre_prob=None,src=None):
        decoder_input_new = decoder_input
        if self.args.copy_decoder:
            tot_ext = decoder_input.max().item()
            if tot_ext >= self.vocab_size:
                decoder_input_new, step,  dec_states_src,dec_states_sent = self.updateExtToken(decoder_input, local_idx2tokenid, src_features,
                dec_states_src, dec_states_sent, sent_repr, mask_sent, mask_word, step)
        probs, dec_out, dec_states_src, attn_context_word = self.wordCopySumm.predict(decoder_input_new, src_features, dec_states_src,step,mask_word,src)

        if self.args.sent_attn:
            _, dec_states_sent, attn_sent = self.decoder_sent(dec_out, sent_repr,
                                                              dec_states_sent, memory_masks=mask_sent,
                                                              step=step)
        else:
            _, dec_states_sent, attn_sent = self.decoder_sent(decoder_input_new, sent_repr,
                                                            dec_states_sent, memory_masks=mask_sent,
                                                            step=step)
        if self.args.copy_decoder:
            attn_src = torch.mean(attn_sent, 1)
            attn_context = torch.matmul(attn_src, sent_repr)
            index = copy_seq
            # p=1
            p = torch.sigmoid(self.p(torch.cat([dec_out.transpose(0, 1).squeeze(0), attn_context_word.squeeze(1),attn_context.squeeze(1)], -1)))
            probs = p * probs
            ext_probs = probs.new_zeros((1, max_node_size)).expand(probs.size(0), -1)
            probs = torch.cat([probs, ext_probs], -1)
            copy_probs = (1 - p) * attn_src.squeeze(1)
            probs = probs.scatter_add(1, index, copy_probs)
        return probs,dec_out,dec_states_src,dec_states_sent,step