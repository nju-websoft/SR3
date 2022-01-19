#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from evaluate import *
from others.util import rouge_results_to_str,tile,test_rouge
from translate.beam import GNMTGlobalScorer
import copy

def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.gpu != '-1'

        self.args = args
        self.model = model
        # self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.split_token = symbols['EOQ']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch['src_tokens'].size(0)
        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch['target'], batch['src_tokens']
        if self.args.split_qm:
            local_idx2token = batch['local_idx2token']
            local_idx2tokenid = batch['local_idx2tokenid']
        translations = []
        for b in range(batch_size):
            if len(preds[b])>0:
                pred_sents=''
                containn=[]
                # id2token=False
                for n in preds[b][0]:
                    n = int(n)
                    if self.args.split_qm and n in local_idx2token[b]:
                        if n not in containn:
                            # id2token=True
                            if self.args.dataset == 'geo':
                                pred_sents+=local_idx2token[b][n]
                            else:
                                # pred_sents += " "+local_idx2token[b][n]
                                pred_sents += " " + " ".join(self.vocab.convert_ids_to_tokens(local_idx2tokenid[b][n]))
                            containn.append(n)
                    else:
                        # if n == self.split_token:
                        #     id2token=False
                        # if not id2token:
                            if self.args.dataset == 'geo':
                                pred_sents+=self.vocab.convert_ids_to_tokens(n)
                            else:
                                pred_sents += " "+self.vocab.convert_ids_to_tokens(n)
                # pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
                # pred_sents = ''.join(pred_sents).replace(' ##','')
                pred_sents = pred_sents.replace(' ##', '')
            else:
                pred_sents = ''
            # print(pred_sents)
            # print(preds[b])
            if self.args.dataset == 'geo':
                gold_sent = ''.join(tgt_str[b]).replace('[unused1]','').replace('[unused3]','').replace('[unused2]','')
            else:
                pred_sents=pred_sents.split('[unused0]')[0]
                gold_sent = ' '.join(tgt_str[b]).replace('[unused1]', '').replace(' [unused3] ', '<q>').replace('[unused2]','').replace(' ##', '')
                gold_sent = re.sub(r' +', ' ', gold_sent)
            # raw_src =''.join([ ''.join([ self.vocab.ids_to_tokens[x] for x in t ]) for t in src[b][1:]]).replace('[PAD]','')
            # print(gold_sent)
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            if self.args.dataset == 'geo':
                raw_src = ''.join(raw_src).replace('[PAD]','')
            else:
                raw_src = ' '.join(raw_src).replace('[PAD]', '').replace(' ##', '')
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        attn_path = self.args.result_path + '.%d.attn.csv' % step
        self.attn_out_file = codecs.open(attn_path, 'w', 'utf-8')
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data=self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans
                    pred_str = pred.replace('[unused1]', '').replace('[unused4]', '').replace('[PAD]',
                                                                                              '').replace(
                        '[unused2]', '').replace(' [unused3] ', '<q>').replace('[unused3]','').strip()
                    pred_str = re.sub(r' +', ' ',pred_str)
                    gold_str = gold.strip()
                    if (self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str + '<q>' + sent.strip()
                            can_gap = math.fabs(len(_pred_str.split()) - len(gold_str.split()))
                            # if(can_gap>=gap):
                            if (len(can_pred_str.split()) >= len(gold_str.split()) + 10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1 and self.args.report_rouge):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges,self.args.dataset)))

            if self.args.dataset=="geo":
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('test/rouge1-F', rouges[("rouge-1", 'f')], step)
                    self.tensorboard_writer.add_scalar('test/rouge2-F', rouges[("rouge-2", 'f')], step)
                    self.tensorboard_writer.add_scalar('test/rougeL-F', rouges[("rouge-l", 'f')], step)
                return rouges[("rouge-1", 'f')]+rouges[("rouge-2", 'f')]+rouges[("rouge-l", 'f')]
            else:
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                    self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                    self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
                return rouges['rouge_1_f_score'] + rouges['rouge_2_f_score'] + rouges['rouge_l_f_score']
        else:
            return 0

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        if self.args.dataset == 'geo':
            results_dict=getScore(can_path,gold_path,'zh')
        else:
            # results_dict = getScore(can_path, gold_path, 'en')
            results_dict = test_rouge(self.args.temp_eval_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                        batch,
                        self.max_length,
                        min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch['src_tokens'].size(0)

        if self.args.split_qm:
            if self.args.train_copy:
                word_vec, words, mask_word = self.model.wordCopySumm.encode_word(batch)
                sent_repr, mask_sent = self.model.encode_step_split(batch)
            else:
                word_vec, words, sent_repr, mask_word, mask_sent = self.model.encode_step_split(batch)
            if self.args.train_copy:
                dec_states = self.model.wordCopySumm.decoder_src.init_decoder_state(words, word_vec, with_cache=True)
                dec_states_sent = self.model.decoder_sent.init_decoder_state(~mask_sent, sent_repr, with_cache=True)
                dec_states_sent.map_batch_fn(
                    lambda state, dim: tile(state, beam_size, dim=dim))
                mask_sent = mask_sent.unsqueeze(1)
                mask_word = mask_word.unsqueeze(1)
            elif self.args.split_gen:
                dec_states = self.model.decoder_src.init_decoder_state(words, word_vec, with_cache=True)
                dec_states_sent = self.model.decoder_sent.init_decoder_state(~mask_sent, sent_repr, with_cache=True)
                dec_states_sent.map_batch_fn(
                    lambda state, dim: tile(state, beam_size, dim=dim))
                mask_sent=mask_sent.unsqueeze(1)
                mask_word = mask_word.unsqueeze(1)
            else:
                dec_states = self.model.decoder.init_decoder_state(words, word_vec, with_cache=True)
            if self.args.copy_decoder:
                local_idx2tokenid = batch['local_idx2tokenid']
                local_idx2token = batch['local_idx2token']
        else:
            if self.args.split_gen:
                top_vec, src, mask_q = self.model.encode_word(batch)
                if self.args.multidoc:
                    dec_states = self.model.decoder.init_decoder_state(src, top_vec, with_cache=True)
                else:
                    dec_states = self.model.decoder_src.init_decoder_state(src, top_vec, with_cache=True)
            else:
                sent_repr, sents = self.model.encode_step(batch)
                dec_states = self.model.decoder.init_decoder_state(sents, sent_repr, with_cache=True)
        device = batch['src_tokens'].device

        # Tile states and memory beam_size times.

        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        if self.args.split_qm:
            src_features = tile(word_vec,beam_size,dim=0)
            src = tile(words, beam_size, dim=0)
            mask_word = tile(mask_word,beam_size,dim=0)
            # if sent_repr!=None:
            sent_repr = tile(sent_repr,beam_size,dim=0)
            mask_sent = tile(mask_sent,beam_size,dim=0)
            if self.args.copy_decoder:
                copy_seq = tile(batch['copy_seq'],beam_size,dim=0)
                local_idx2tokenid_new =[]
                for x in local_idx2tokenid:
                    for i in range(beam_size):
                        local_idx2tokenid_new.append(x)
                local_idx2token_new = []
                for x in local_idx2token:
                    for i in range(beam_size):
                        local_idx2token_new.append(x)
        else:
            if self.args.split_gen:
                src_features = tile(top_vec, beam_size, dim=0)
                src = tile(src, beam_size, dim=0)
                mask_word = tile(mask_q, beam_size, dim=0)
            else:
                src_features = tile(sent_repr, beam_size, dim=0)
                src = tile(sents, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
                [batch_size * beam_size, 1],
                self.start_token,
                dtype=torch.long,
                device=device)

        # print(alive_seq)
        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812
        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch
        steps = torch.zeros_like(alive_seq).squeeze(-1)
        copy_step = [0 for _ in range(batch_size*beam_size)]
        pre_prob = None
        for step in range(max_length):
            # print(alive_seq)
            # print(copy_step)
            decoder_input = alive_seq[:, -1].view(1, -1)
            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            if self.args.split_qm:
                if self.args.split_gen or self.args.train_copy:
                    if self.args.copy_decoder:
                        probs, dec_out, dec_states, dec_states_sent, steps = self.model.predict_split_qm(decoder_input, src_features,
                                                                                        dec_states,dec_states_sent,
                                                                                        sent_repr, mask_sent, mask_word,
                                                                                        steps, copy_seq,
                                                                                        local_idx2tokenid_new, src=src)
                        steps += 1
                        pre_prob = probs
                    else:
                        probs, dec_out, dec_states, _ = self.model.predict_split_qm(decoder_input, src_features, dec_states,
                                                                                    sent_repr, mask_sent, mask_word, step,
                                                                                    src=src)
                else:
                    if self.args.copy_decoder:
                        probs, dec_out, dec_states,steps = self.model.predict_split_qm(decoder_input, src_features, dec_states,
                                                                                 sent_repr, mask_sent, mask_word, steps,copy_seq,local_idx2tokenid_new,src=src)
                        steps+=1
                        pre_prob = probs
                    else:
                        probs, dec_out, dec_states,_ =self.model.predict_split_qm(decoder_input, src_features, dec_states, sent_repr, mask_sent, mask_word, step,src=src)
            else:
                if self.args.split_gen:# and self.args.copy_decoder:
                    probs, dec_out, dec_states = self.model.predict(decoder_input, src_features, dec_states, step,mask_word,src)
                else:
                    probs, dec_out, dec_states= self.model.predict(decoder_input, src_features, dec_states, step)
            # print(probs.topk(probs.size(0), dim=-1))
            # print(probs.size())

            log_probs = torch.log(probs)

            # if (step > 0):
            #     for i in range(alive_seq.size(0)):
            #         if copy_step[i] == 0:
            #             words = [int(w) for w in alive_seq[i]]
            #             ext_word = set([word for word in words if word >= self.model.vocab_size])
            #             for w in ext_word:
            #                 log_probs[i][w] = -1e20

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if(self.args.block_trigram):
                cur_len = alive_seq.size(1)
                # for i in range(alive_seq.size(0)):
                #     print([self.vocab.convert_ids_to_tokens(int(word)) if word<self.model.vocab_size else local_idx2token_new[i][int(word)] for word in alive_seq[i] ])
                if (cur_len > 1):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        trigrams = [words[i] for i in range(1, len(words) - 2) if words[i]>=self.model.vocab_size]
                        trigram = words[-1]
                        if trigram in trigrams:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i] if w < self.model.vocab_size]
                        # words = [self.vocab.ids_to_tokens[w] for w in words]
                        # words = ' '.join(words).replace(' ##','').split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # print(topk_beam_index)
            # print(beam_offset)
            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            # print(batch_index)
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            # print(select_indices)
            src_features = src_features.index_select(0, select_indices)
            src = src.index_select(0,select_indices)
            if self.args.split_qm:

                mask_word = mask_word.index_select(0,select_indices)
                # if sent_repr != None:
                sent_repr = sent_repr.index_select(0, select_indices)
                mask_sent = mask_sent.index_select(0,select_indices)
                steps = steps.index_select(0, select_indices)
                if self.args.copy_decoder:
                    pre_prob = pre_prob.index_select(0,select_indices)
                    copy_seq = copy_seq.index_select(0,select_indices)
                    select_indices_lists = select_indices.cpu().numpy().tolist()
                    temp = local_idx2tokenid_new
                    local_idx2tokenid_new = []
                    for i in select_indices_lists:
                        local_idx2tokenid_new.append(temp[i])
                    temp = local_idx2token_new
                    local_idx2token_new = []
                    for i in select_indices_lists:
                        local_idx2token_new.append(temp[i])
                    temp = copy_step
                    copy_step = []
                    for i in select_indices_lists:
                        copy_step.append(temp[i])
            else:
                if self.args.split_gen:
                    # src_features = src_features.index_select(0, select_indices)
                    mask_word = mask_word.index_select(0, select_indices)
                    # src = src.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))
            if self.args.split_gen and self.args.split_qm:
                dec_states_sent.map_batch_fn(
                    lambda state, dim: state.index_select(dim, select_indices))
            # if steps.min(-1)[0].cpu().item()>max_length:
            #     break
        return results

class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
