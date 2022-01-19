import sys
import os
curpath=os.path.abspath(os.path.dirname(__file__))
rootpath=os.path.split(curpath)[0]
sys.path.append(rootpath)
import random
import torch
from torch import nn
from others.logging import *
import numpy as np
from model_graph.extract import read_file,read_cnndm_file
from model_graph.utils import move_to_device
from model_graph.tokenizer_graph import BertData
import gc
PAD, UNK = '[PAD]', '[UNK]'
CLS = '[CLS]'
STR, END = '[unused1]', '[unused2]'
SPLIT = '[unused3]'
SEL, rCLS, TL = '[SELF]', '[rCLS]', '[TL]'

class Vocab(object):
    def __init__(self,specials):
        idx2token = [PAD, UNK] +specials
        self._priority = dict()
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def _back_to_txt_for_check(tensor, vocab, local_idx2token=None):
    for bid, xs in enumerate(tensor.t().tolist()):
        txt = []
        for x in xs:
            if x == vocab.padding_idx:
                break
            if x >= vocab.size:
                assert local_idx2token is not None
                assert local_idx2token[bid] is not None
                tok = local_idx2token[bid][x]
            else:
                tok = vocab.idx2token(x)
            txt.append(tok)
        txt = ' '.join(txt)
        print (txt)

def ListsToTensor(xs, tokenizer=None,vocab=None):
    def toIdx(w, i):
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if vocab and w in vocab[i]:
            return vocab[i][w]
        if tokenizer is None:
            return w
        else:
            return tokenizer.tokenizer.convert_tokens_to_ids(w)
    pad = tokenizer.pad_vid if tokenizer else 0
    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x,i) + [pad]*(max_len-len(x))
        ys.append(y)
    # data = np.transpose(np.array(ys))
    data = np.array(ys)
    return data

def ListsofSentToTensor(xs, tokenizer=None, max_string_len=360):
    max_len = max(len(x) for x in xs)
    max_item_len = max(max(len(y) for y in x) for x in xs)
    # if(max_item_len>max_string_len):
    #     print(max_item_len)
    max_string_len = min(max_string_len,max_item_len)
    ys = []
    for x in xs:
        if tokenizer:
            y = x + [PAD]*(max_len -len(x))
            zs = []
            for z in y:
                z = z.split()[:max_string_len]
                zs.append(tokenizer.tokenizer.convert_tokens_to_ids(z) + [tokenizer.pad_vid]*(max_string_len - len(z)))
        else:
            y = x + [[0]] * (max_len - len(x))
            zs = []
            for z in y:
                z = z[:max_string_len]
                zs.append(
                    z + [0] * (max_string_len - len(z)))
        ys.append(zs)
    data = np.array(ys)
    # data = np.transpose(np.array(ys), (1, 0, 2))
    return data

def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([ list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
        #tensor = torch.from_numpy(data).long()
    return data


def ListsToTensorCls(xs):
    max_len = max(len(x) for x in xs)
    ys = []
    mask_y = []
    for i, x in enumerate(xs):
        y = x + [0]*(max_len-len(x))
        mask_y.append([0]*len(x)+[1]*(max_len-len(x)))
        ys.append(y)
    # data = np.transpose(np.array(ys))
    data_y = np.array(ys)
    data_mask_y = np.array(mask_y)
    return data_y,data_mask_y

def batchify(data, vocabs,tokenizer,device,args):
    tgt_subtokens,sents,tgt_subtokens_out,raw_tgt_subtokens,src_tokens,segments,cls_ids=[],[],[],[],[],[],[]
    sep_ids = []

    if args.split_qm and args.graph_transformer:

        for x in data:
            tgt_subtoken, sent, tgt_subtoken_out, raw_tgt_subtoken, src_token, segment,cls_id = x['tgt_subtoken'],x['sents_pre'],x['tgt_subtokens_out'],x['raw_tgt_subtokens_str'],x['src_tokens'],x['segments'],x['cls_ids']
            # tgt_subtoken,sent,tgt_subtoken_out,raw_tgt_subtoken,src_token,segment=tokenizer.preprocess(x)

            tgt_subtokens.append(tgt_subtoken)
            sents.append(sent)
            tgt_subtokens_out.append(tgt_subtoken_out)
            raw_tgt_subtokens.append(raw_tgt_subtoken)
            src_tokens.append(src_token)
            segments.append(segment)
            cls_ids.append(cls_id)
            sep_ids.append(cls_id[1:]+[len(src_token)])

        sents_token = ListsofSentToTensor([[CLS] + [x['tokens'] for x in sent] for sent in sents], tokenizer,max_string_len=args.max_string_len)
        # temp = [[[0]] + [x['segments'] for x in sent] for sent in sents]
        sents_segments = ListsofSentToTensor([[[0]] + [x['segments'] for x in sent] for sent in sents],max_string_len=args.max_string_len)
    else:
        for x in data:
            tgt_subtoken, tgt_subtoken_out, raw_tgt_subtoken, src_token, segment, cls_id = x['tgt_subtoken'], x[
                'tgt_subtokens_out'], x['raw_tgt_subtokens_str'], x['src_tokens'], x['segments'], x['cls_ids']
            # tgt_subtoken,sent,tgt_subtoken_out,raw_tgt_subtoken,src_token,segment=tokenizer.preprocess(x)
            tgt_subtokens.append(tgt_subtoken)
            tgt_subtokens_out.append(tgt_subtoken_out)
            raw_tgt_subtokens.append(raw_tgt_subtoken)
            src_tokens.append(src_token)
            segments.append(segment)
            cls_ids.append(cls_id)
            sep_ids.append(cls_id[1:] + [len(src_token)])
    cls_ids, cls_mask = ListsToTensorCls(cls_ids)
    sep_ids, _ = ListsToTensorCls(sep_ids)

    tgt = ListsToTensor(tgt_subtokens,tokenizer)
    src_tokens = ListsToTensor(src_tokens,tokenizer)
    segments = ListsToTensor(segments)

    if args.split_qm:

        local_token2idx = [x['token2idx'] for x in data]
        local_idx2token = [{int(k):v for k,v in  x['idx2token'].items()} for x in data]
        local_idx2tokenid = [{int(k):v for k,v in  x['idx2tokenid'].items()} for x in data]

        tgt_out = ListsToTensor(tgt_subtokens_out,tokenizer,local_token2idx)

        if args.graph_transformer:
            _depth = ListsToTensor([[0] + x['depth'] for x in data])
            all_relations = dict()
            cls_idx = vocabs['relation'].token2idx(CLS)
            rcls_idx = vocabs['relation'].token2idx(rCLS)
            self_idx = vocabs['relation'].token2idx(SEL)
            all_relations[tuple([cls_idx])] = 0
            all_relations[tuple([rcls_idx])] = 1
            all_relations[tuple([self_idx])] = 2


            _relation_type = []
            for bidx, x in enumerate(data):
                x['relation'] = {int(k):{int(k1):v1 for k1,v1 in v.items()} for k,v in x['relation'].items()}
                n = len(x['sents'])
                brs = [ [2]+[0]*(n) ]
                for i in range(n):
                    rs = [1]
                    for j in range(n):
                        all_path = x['relation'][i][j]
                        path = random.choice(all_path)['edge']
                        if len(path) == 0: # self loop
                            path = [SEL]
                        if len(path) > 8: # too long distance
                            path = [TL]
                        path = tuple(vocabs['relation'].token2idx(path))
                        rtype = all_relations.get(path, len(all_relations))
                        if rtype == len(all_relations):
                            all_relations[path] = len(all_relations)
                        rs.append(rtype)
                    rs = np.array(rs, dtype=np.int)
                    brs.append(rs)
                brs = np.stack(brs)
                _relation_type.append(brs)
            # _relation_type = ArraysToTensor(_relation_type)
            _relation_type = np.transpose(ArraysToTensor(_relation_type), (2, 1, 0))
            # _relation_bank[_relation_type[i][j][b]] => from j to i go through what

            B = len(all_relations)
            _relation_bank = dict()
            _relation_length = dict()
            for k, v in all_relations.items():
                _relation_bank[v] = np.array(k, dtype=np.int)
                _relation_length[v] = len(k)
            _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
            _relation_length = [_relation_length[i] for i in range(len(all_relations))]
            _relation_bank = np.transpose(ArraysToTensor(_relation_bank))
            # _relation_bank = ArraysToTensor(_relation_bank)
            _relation_length = np.array(_relation_length)

        _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocab=local_token2idx)

        if not args.copy_decoder:
            token_out = tgt[:,1:]
        else:
            token_out = tgt_out[:,1:]

        if args.graph_transformer:
            if args.use_cls:
                ret = {
                    # 'sents': move_to_device(sents_token,device),
                    # 'sents_segments': move_to_device(sents_segments, device),
                    # 'sent_depth': move_to_device(_depth,device),
                    'relation': move_to_device(_relation_type,device),
                    'relation_bank': move_to_device(_relation_bank,device),
                    'relation_length': move_to_device(_relation_length,device),
                    'local_idx2token': local_idx2token,
                    'local_token2idx': local_token2idx,
                    'local_idx2tokenid' : local_idx2tokenid,
                    'token_in':move_to_device(tgt[:,:-1],device),
                    'token_out':move_to_device(token_out,device),
                    'token_gen':move_to_device(tgt[:,1:],device),
                    # 'target':tgt_subtokens,
                    'target': raw_tgt_subtokens,
                    'copy_seq':move_to_device(_cp_seq,device),
                    'src_tokens':move_to_device(src_tokens,device),
                    'segments':move_to_device(segments,device),
                    'cls_ids': move_to_device(cls_ids, device),
                    'sep_ids': move_to_device(sep_ids, device),
                    'mask_cls': move_to_device(cls_mask, device)
                }
            else:
                ret = {
                    'sents': move_to_device(sents_token,device),
                    'sents_segments': move_to_device(sents_segments, device),
                    'sent_depth': move_to_device(_depth,device),
                    'relation': move_to_device(_relation_type,device),
                    'relation_bank': move_to_device(_relation_bank,device),
                    'relation_length': move_to_device(_relation_length,device),
                    'local_idx2token': local_idx2token,
                    'local_token2idx': local_token2idx,
                    'local_idx2tokenid' : local_idx2tokenid,
                    'token_in':move_to_device(tgt[:,:-1],device),
                    'token_out':move_to_device(token_out,device),
                    'token_gen':move_to_device(tgt[:,1:],device),
                    # 'target':tgt_subtokens,
                    'target': raw_tgt_subtokens,
                    'copy_seq':move_to_device(_cp_seq,device),
                    'src_tokens':move_to_device(src_tokens,device),
                    'segments':move_to_device(segments,device)
                }
        else:
            ret = {
                'local_idx2token': local_idx2token,
                'local_token2idx': local_token2idx,
                'local_idx2tokenid' : local_idx2tokenid,
                'token_in':move_to_device(tgt[:,:-1],device),
                'token_out':move_to_device(token_out,device),
                'token_gen':move_to_device(tgt[:,1:],device),
                # 'target':tgt_subtokens,
                'target': raw_tgt_subtokens,
                'copy_seq':move_to_device(_cp_seq,device),
                'src_tokens':move_to_device(src_tokens,device),
                'segments':move_to_device(segments,device),
                'cls_ids': move_to_device(cls_ids, device),
                'sep_ids': move_to_device(sep_ids, device),
                'mask_cls': move_to_device(cls_mask, device)
            }
    else:
        ret = {
            # 'sents': move_to_device(sents, device),
            'token_in': move_to_device(tgt[:, :-1], device),
            'token_gen': move_to_device(tgt[:, 1:], device),
            'token_out': move_to_device(tgt[:, 1:], device),
            # 'target':tgt_subtokens,
            'target': raw_tgt_subtokens,
            'src_tokens': move_to_device(src_tokens, device),
            'segments': move_to_device(segments, device)
        }
        # print(ret['sents'].size(),ret['token_in'].size(),ret['token_gen'].size(),ret['token_out'].size(),ret['src_tokens'].size(),ret['segments'].size())

    if args.classify:
        nodes_labels = [x['node_labels'] for x in data]
        nodes_labels, _ = ListsToTensorCls(nodes_labels)
        ret['node_labels'] = move_to_device(nodes_labels,device)
    return ret


def abs_batch_size_fn(new, count):
    src, tgt = new['src_tokens'], new['tgt_subtoken']
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt) + len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements



def load_train_dataset(args):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.
    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    import glob

    def _lazy_dataset_loader(pt_file):
        dataset = torch.load(pt_file)
        logger.info('Loading dataset from %s, number of examples: %d' %
                    (pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path  + 'train_graph[0-9]*.0.bert'))
    print(pts)
    if pts:
        random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + 'train_graph.bert'
        yield _lazy_dataset_loader(pt)


class DataloaderWikihow(object):
    def __init__(self, args,vocabs,tokenizer,data, batch_size,device, for_train):
        self.args = args
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.vocabs=vocabs
        self.tokenizer=tokenizer
        self.for_train=for_train
        self.cur_iter = self._next_dataset_iterator(data)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.data)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None
        return DataLoader(self.args,self.vocabs,self.tokenizer,self.cur_dataset, self.batch_size,self.device, self.for_train)


class DataLoader(object):
    def __init__(self, args,vocabs,tokenizer,data, batch_size,device, for_train):
        self.data = data
        self.args = args
        self.tokenizer=tokenizer
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.
        self.nprocessors = 8
        self.device = device
        self.record_flag = False
        self.batch_size_fn = abs_batch_size_fn


    def set_unk_rate(self, x):
        self.unk_rate = x

    def record(self):
        self.record_flag = True

    # def datalen(self,x):
    #     return len(x['sents'])** 2 + len(x['tgt'])

    def datalen(self, x):
        # print(len(x['src_tokens']))
        # print(len(x['tgt_subtoken']))
        return len(x['src_tokens']) + len(x['tgt_subtoken'])
    # def __iter__(self):
    #     idx = list(range(len(self.data)))
    #     print(len(self.data))
    #     for d in self.data:
    #         yield d
    def __iter__(self):
    # def count(self):
        idx = list(range(len(self.data)))
        # print(len(self.data))
        if self.train:
            random.shuffle(idx)
            # idx.sort(key=lambda x: len(self.data[x]['tgt_subtoken'])+len(self.data[x]['src_tokens'])*2)
            # idx.sort(key=lambda x: len(self.data[x]['src_tokens']))
            # idx.sort(key = lambda x: len(self.data[x]))

            idx.sort(key=lambda x: self.datalen(self.data[x]))

        batches = []
        num_tokens, batch = 0, []

        # batch, size_so_far = [], 0
        for i in idx:
            num_tokens += self.datalen(self.data[i])
            # print(self.data[i],num_tokens)
            batch.append(self.data[i])

            # size_so_far = self.batch_size_fn(self.data[i], len(batch))
            # if size_so_far == self.batch_size:
            #     batches.append(batch)
            #     batch, size_so_far = [], 0
            # elif size_so_far > self.batch_size:
            #     if len(batch)>1:
            #         batches.append(batch[:-1])
            #     batch, size_so_far = batch[-1:], self.batch_size_fn(self.data[i], 1)

            if num_tokens >= self.batch_size or len(batch)>256:
                batches.append(batch)
                num_tokens, batch = 0, []

        # if not self.train or num_tokens > self.batch_size/2:
        #     if len(batch)>0:
        #         batches.append(batch)
        if batch:
            batches.append(batch)

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            # res = [work(g) for g in batch]
            res=batch
            if not self.record_flag:
                yield batchify(res, self.vocabs,self.tokenizer,self.device,self.args)
            else:
                yield batchify(res, self.vocabs, self.tokenizer,self.device,self.args), res


def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, default='graph_data/test_graph.json')
    parser.add_argument('--bert_data', type=str, default='graph_data/test_graph.bert')
    parser.add_argument('--train_batch_size', type=int, default=500)
    parser.add_argument("-encoder_name", default='bert-base-chinese', type=str)
    parser.add_argument("-max_string_len", default=300, type=int)
    parser.add_argument('-max_pos', type=int, default=512)
    parser.add_argument('-max_tgt_len', type=int, default=200)
    parser.add_argument("-copy_decoder", default=True, type=bool)
    parser.add_argument("-split_qm", default=True, type=bool)
    parser.add_argument("-comfirm_connect", default=True, type=bool)
    parser.add_argument('-max_node_size', type=int, default=5)
    parser.add_argument('-min_copy_rouge', type=float, default=0.3)
    parser.add_argument('-use_rouge_f', type=bool, default=True)
    parser.add_argument('-recovery_order', type=bool, default=True)
    parser.add_argument('-use_cls', type=bool, default=False)
    parser.add_argument('-classify', type=bool, default=False)

    parser.add_argument("-dataset", default='geo', type=str,choices=['geo','wikihow','cnndm'])
    return parser.parse_args()

if __name__ == '__main__':
    from model_graph.extract import LexicalMap
    import time
    args = parse_config()

    args.dataset = 'geo'
    args.use_cls = True
    args.use_rouge_f = False
    args.max_node_size = 7
    args.min_copy_rouge = 0.3
    args.encoder_name = 'bert-base-chinese'
    vocabs = dict()
    tokenizer = BertData(args)
    vocabs['tokens'] = tokenizer.tokenizer.vocab
    vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity'] + [CLS, rCLS, SEL, TL])
    for mode in ["train","val","test"]:
        args.raw_data = "../data/" + mode + "_graph.json"
        args.bert_data = "../data/" + mode + "_graph.bert"
        lexical_mapping = LexicalMap()
        train_data = read_file(args.raw_data, args, tokenizer, lexical_mapping)
        torch.save(train_data, args.bert_data)