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

def ListsofSentToTensor(xs, tokenizer=None, max_string_len=512,max_doc_size=4):
    max_len = max(len(x) for x in xs)
    max_len = min(max_doc_size,max_len)
    max_item_len = max(max(len(y) for y in x[:max_len]) for x in xs)
    # if(max_item_len>max_string_len):
    #     print(max_item_len)
    max_string_len = min(max_string_len,max_item_len)
    ys = []
    for x in xs:
        if tokenizer:
            y = x[:max_len] + [[PAD]]*(max_len -len(x))
            zs = []
            for z in y:
                z = z[:max_string_len]
                zs.append(tokenizer.tokenizer.convert_tokens_to_ids(z) + [tokenizer.pad_vid]*(max_string_len - len(z)))
        else:
            y = x[:max_len] + [[0]] * (max_len - len(x))
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
    tgt_subtokens,tgt_subtokens_out,src_tokens,segments,cls_ids,node_labels=[],[],[],[],[],[]
    sep_ids = []

    if args.split_qm and args.graph_transformer:

        for x in data:
            tgt_subtoken, tgt_subtoken_out, src_token, segment,cls_id,sep_id,node_label = x['tgt_subtoken'],\
              x['tgt_subtokens_out'],x['src_tokens'],x['segments'],x['cls_ids'],x['sep_ids'],x['node_labels']
            # tgt_subtoken,sent,tgt_subtoken_out,raw_tgt_subtoken,src_token,segment=tokenizer.preprocess(x)

            tgt_subtokens.append(tgt_subtoken)
            tgt_subtokens_out.append(tgt_subtoken_out)
            src_tokens.append(src_token)
            segments.append(segment)
            cls_ids.append(cls_id)
            sep_ids.append(sep_id)
            node_labels.append(node_label)
    else:
        for x in data:
            tgt_subtoken, tgt_subtoken_out, src_token, segment, cls_id,sep_id,node_label = x['tgt_subtoken'], x[
                'tgt_subtokens_out'], x['src_tokens'], x['segments'], x['cls_ids'],x['sep_ids'],x['node_labels']
            # tgt_subtoken,sent,tgt_subtoken_out,raw_tgt_subtoken,src_token,segment=tokenizer.preprocess(x)
            tgt_subtokens.append(tgt_subtoken)
            tgt_subtokens_out.append(tgt_subtoken_out)
            src_tokens.append(src_token)
            segments.append(segment)
            cls_ids.append(cls_id)
            sep_ids.append(sep_id)
            node_labels.append(node_label)
    src_tokens = ListsofSentToTensor(src_tokens, tokenizer,args.max_pos,args.max_doc_size)
    segments = ListsofSentToTensor(segments,None,args.max_pos,args.max_doc_size)
    # batch,doc_size,doc_len
    doc_len = src_tokens.shape[2]
    doc_size = src_tokens.shape[1]
    new_cls_ids = []
    new_sep_ids = []
    new_node_labels = []

    # relation_idxs = []
    assert len(cls_ids)==len(node_labels)
    assert len(cls_ids)==len(sep_ids)
    for cls_id,sep_id,node_label in zip(cls_ids,sep_ids,node_labels):
        assert len(cls_id) == len(node_label)
        assert len(cls_id) == len(sep_id)
        new_cls_id=[]
        new_sep_id=[]
        new_node_label=[]
        # relation_idx=[]
        prefix = 0
        present = 0
        for idx,(doc_cls_id,doc_sep_id,doc_node_label) in enumerate(zip(cls_id,sep_id,node_label)):
            assert len(doc_cls_id) == len(doc_node_label)
            assert len(doc_cls_id) == len(doc_sep_id)
            new_cls_id.extend([t+prefix for t in doc_cls_id])
            new_sep_id.extend([t+prefix for t in doc_sep_id])
            # relation_idx.extend([0]+[i+present+1 for i in range(len(doc_cls_id)-1)])
            new_node_label.extend(doc_node_label)
            if idx>=doc_size:
                break
            prefix += doc_len
            present +=(len(doc_cls_id)-1)
        new_cls_ids.append(new_cls_id)
        new_sep_ids.append(new_sep_id)
        new_node_labels.append(new_node_label)
        # relation_idxs.append(relation_idx)

    # cls_ids = new_cls_ids
    # sep_ids = new_sep_ids
    cls_ids, cls_mask = ListsToTensorCls(new_cls_ids)
    sep_ids, _ = ListsToTensorCls(new_sep_ids)
    node_labels = ListsToTensor(new_node_labels)

    tgt = ListsToTensor(tgt_subtokens,tokenizer)

    if args.split_qm:

        local_token2idx = [x['token2idx'] for x in data]
        local_idx2token = [{int(k):v for k,v in  x['idx2token'].items()} for x in data]
        local_idx2tokenid = [{int(k):v for k,v in  x['idx2tokenid'].items()} for x in data]

        tgt_out = ListsToTensor(tgt_subtokens_out,tokenizer,local_token2idx)

        if args.graph_transformer:
            # _depth = ListsToTensor([[0] + x['depth'] for x in data])
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
                # relation_idx = relation_idxs[bidx]
                brs = [ [2]+[0]*n ]
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
            _relation_type = ArraysToTensor(_relation_type)
            _relation_type = np.transpose(_relation_type, (2, 1, 0))
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
            # if args.use_cls:
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
                    'target':tgt_subtokens,
                    'copy_seq':move_to_device(_cp_seq,device),
                    'src_tokens':move_to_device(src_tokens,device),
                    'segments':move_to_device(segments,device),
                    'cls_ids': move_to_device(cls_ids, device),
                    'node_labels': move_to_device(node_labels, device),
                    'sep_ids': move_to_device(sep_ids, device),
                    'mask_cls': move_to_device(cls_mask, device)
                }
        else:
            ret = {
                'local_idx2token': local_idx2token,
                'local_token2idx': local_token2idx,
                'local_idx2tokenid' : local_idx2tokenid,
                'token_in':move_to_device(tgt[:,:-1],device),
                'token_out':move_to_device(token_out,device),
                'token_gen':move_to_device(tgt[:,1:],device),
                'target':tgt_subtokens,
                'copy_seq':move_to_device(_cp_seq,device),
                'src_tokens':move_to_device(src_tokens,device),
                'segments':move_to_device(segments,device),
                'cls_ids': move_to_device(cls_ids, device),
                'node_labels': move_to_device(node_labels, device),
                'sep_ids': move_to_device(sep_ids, device),
                'mask_cls': move_to_device(cls_mask, device)
            }
    else:
        ret = {
            # 'sents': move_to_device(sents, device),
            'token_in': move_to_device(tgt[:, :-1], device),
            'token_gen': move_to_device(tgt[:, 1:], device),
            'token_out': move_to_device(tgt[:, 1:], device),
            'target':tgt_subtokens,
            'src_tokens': move_to_device(src_tokens, device),
            'segments': move_to_device(segments, device)
        }
        # print(ret['sents'].size(),ret['token_in'].size(),ret['token_gen'].size(),ret['token_out'].size(),ret['src_tokens'].size(),ret['segments'].size())
    # if ret['cls_ids'].size(1)==0:
    #     print()
    return ret


def abs_batch_size_fn(new, count):
    src, tgt = new['src_tokens'], new['tgt_subtoken']
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    maxsrc=max([len(item) for item in src])
    max_n_sents = max(max_n_sents, len(tgt)+maxsrc*len(src))
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
    pts = sorted(glob.glob(args.data_path  + 'train_graph[0-9]*.bert'))
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

    def datalen(self, x):
        # print(x)
        maxsrc = max([len(item) for item in x['src_tokens']])
        return len(x['tgt_subtoken']) + maxsrc * min(self.args.max_doc_size,len(x['src_tokens']))

    def __iter__(self):
    # def count(self):
        idx = [i for i,d in enumerate(self.data) if len(d['src_tokens'])>0]
        if self.train:
            random.shuffle(idx)
            idx.sort(key=lambda x: self.datalen(self.data[x]))

        batches = []
        num_tokens, batch = 0, []

        # batch, size_so_far = [], 0
        for i in idx:
            num_tokens += self.datalen(self.data[i])
            # print(self.data[i],num_tokens)
            batch.append(self.data[i])

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
    parser.add_argument('-max_doc_size', type=int, default=4)
    parser.add_argument('-min_copy_rouge', type=float, default=0.3)
    parser.add_argument('-use_rouge_f', type=bool, default=True)
    parser.add_argument('-recovery_order', type=bool, default=True)
    parser.add_argument('-use_cls', type=bool, default=False)

    parser.add_argument("-dataset", default='geo', type=str,choices=['geo','wikihow','cnndm'])
    return parser.parse_args()

if __name__ == '__main__':
    from extract import LexicalMap
    import time
    args = parse_config()

    # args.dataset = 'geo'
    # args.use_cls = True
    # args.max_node_size=6
    # args.encoder_name = 'bert-base-chinese'
    # # args.raw_data = 'image_data_20201226/json_data/test_graph.json'
    # # args.bert_data = 'image_data_20201226/json_data/test_graph.bert'
    # args.raw_data = 'image_data_20201226/json_data/val_graph.json'
    # args.bert_data = 'image_data_20201226/bert_data/val_graph.bert'
    # # args.raw_data = 'image_data_20201226/json_data/train_graph.json'
    # # args.bert_data = 'image_data_20201226/bert_data/train_graph.bert'
    # vocabs = dict()
    # tokenizer = BertData(args)
    # vocabs['tokens'] = tokenizer.tokenizer.vocab
    # vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity'] + [CLS, rCLS, SEL, TL])
    # import json
    # lexical_mapping = LexicalMap()
    # train_data = read_file(args.raw_data, args, tokenizer, lexical_mapping)
    # torch.save(train_data, args.bert_data)

    # args.dataset = 'geo'
    # args.use_cls = True
    # args.max_node_size=1
    # args.encoder_name = 'bert-base-chinese'
    # # args.raw_data = 'image_data_20201226/json_data/test_graph.json'
    # # args.bert_data = 'image_data_20201226/bert_data_q/test_graph.bert'
    # # args.raw_data = 'image_data_20201226/json_data/val_graph.json'
    # # args.bert_data = 'image_data_20201226/bert_data_q/val_graph.bert'
    # args.raw_data = 'image_data_20201226/json_data/train_graph.json'
    # args.bert_data = 'image_data_20201226/bert_data_q/train_graph.bert'
    # vocabs = dict()
    # tokenizer = BertData(args)
    # vocabs['tokens'] = tokenizer.tokenizer.vocab
    # vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity'] + [CLS, rCLS, SEL, TL])
    # import json
    # lexical_mapping = LexicalMap()
    # train_data = read_file(args.raw_data, args, tokenizer, lexical_mapping)
    # torch.save(train_data, args.bert_data)

    # args.dataset = 'cnndm'
    # args.max_node_size = 20
    # args.encoder_name = 'bert-base-uncased'
    # # args.raw_data = 'cnndm_data/json_data/test.json'
    # # args.bert_data = 'cnndm_data/bert_data/test_graph.bert'
    # # args.raw_data = 'cnndm_data/json_data/val.json'
    # # args.bert_data = 'cnndm_data/bert_data/val_graph.bert'
    # args.raw_data = 'cnndm_data/json_data/train.json'
    # args.bert_data = 'cnndm_data/bert_data/train_graph.bert'
    # args.use_rouge_f = False
    # args.split_qm = False
    # args.copy_decoder = False

    # args.raw_data = 'cnndm_data/json_data/test.json'
    # args.bert_data = 'cnndm_data/bert_data/sent_copy/test_graph.bert'
    # args.raw_data = 'cnndm_data/json_data/val.json'
    # args.bert_data = 'cnndm_data/bert_data/sent_copy/val_graph.bert'
    # args.raw_data = 'cnndm_data/json_data/train.json'
    # args.bert_data = 'cnndm_data/bert_data/sent_copy/train_graph.bert'
    # args.use_rouge_f = False
    # vocabs = dict()
    # tokenizer = BertData(args)
    # vocabs['tokens'] = tokenizer.tokenizer.vocab

    # vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity','not_connect'] + [CLS, rCLS, SEL, TL])

    import json

    # lexical_mapping = LexicalMap()
    # train_data = read_cnndm_file(args.raw_data, args, tokenizer, lexical_mapping)
    # torch.save(train_data, args.bert_data)


    # args.dataset = 'wikihow'
    # args.max_node_size=20
    # args.encoder_name = 'bert-base-uncased'
    # # args.raw_data = 'wikihow_data/json_data/test_graph.json'
    # # args.bert_data = 'wikihow_data/bert_data/test_graph_recovery_order_seg01.bert'
    # # args.raw_data = 'wikihow_data/json_data/val_graph.json'
    # # args.bert_data = 'wikihow_data/bert_data/val_graph_recovery_order_seg01.bert'
    # args.raw_data = 'wikihow_data/json_data/train_graph.json'
    # args.bert_data = 'wikihow_data/bert_data/train_graph_recovery_order_seg0.bert'
    # args.use_rouge_f = False
    # vocabs = dict()
    # tokenizer = BertData(args)
    # vocabs['tokens'] = tokenizer.tokenizer.vocab
    # vocabs['relation'] = Vocab(
    #     ['q_senario', 'senario_q', 'samedoc'] + [CLS, rCLS, SEL, TL])
    #
    # # vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity','not_connect'] + [CLS, rCLS, SEL, TL])
    #
    # import json
    #
    # lexical_mapping = LexicalMap()
    # train_data = read_file(args.raw_data, args, tokenizer, lexical_mapping)
    # torch.save(train_data, args.bert_data)

    # args.dataset = 'wikihow'
    # args.max_node_size = 20
    # args.min_copy_rouge = 0.4
    # args.encoder_name = 'bert-base-uncased'
    # # args.raw_data = 'wikihow_data/json_data/test_graph.json'
    # # args.bert_data = 'wikihow_data/bert_data/train_graph_recovery_order_0.4/test_graph.bert'
    # # args.raw_data = 'wikihow_data/json_data/val_graph.json'
    # # args.bert_data = 'wikihow_data/bert_data/train_graph_recovery_order_0.4/val_graph.bert'
    # args.raw_data = 'wikihow_data/json_data/train_graph.json'
    # args.bert_data = 'wikihow_data/bert_data/train_graph_recovery_order_0.4/train_graph.bert'
    # args.use_rouge_f = False
    # vocabs = dict()
    # tokenizer = BertData(args)
    # vocabs['tokens'] = tokenizer.tokenizer.vocab
    # vocabs['relation'] = Vocab(
    #     ['q_senario', 'senario_q', 'samedoc'] + [CLS, rCLS, SEL, TL])
    #
    # import json
    #
    # lexical_mapping = LexicalMap()
    # train_data = read_file(args.raw_data, args, tokenizer, lexical_mapping)
    # torch.save(train_data, args.bert_data)

    # args.dataset = 'wikihow'
    # args.max_node_size = 20
    # args.encoder_name = 'bert-base-uncased'
    # # args.raw_data = 'wikihow_data/json_data/test.json'
    # # args.bert_data = 'wikihow_data/bert_data/not_rank/test_graph.bert'
    # # args.raw_data = 'wikihow_data/json_data/val.json'
    # # args.bert_data = 'wikihow_data/bert_data/not_rank/val_graph.bert'
    # args.raw_data = 'wikihow_data/json_data/train.json'
    # args.bert_data = 'wikihow_data/bert_data/not_rank/train_graph.bert'
    # args.use_rouge_f = False
    # args.split_qm = False
    # args.copy_decoder = False
    # vocabs = dict()
    # tokenizer = BertData(args)
    # vocabs['tokens'] = tokenizer.tokenizer.vocab
    #
    # # vocabs['relation'] = Vocab(['q_senario', 'senario_q', 'samedoc', 'sameEntity','not_connect'] + [CLS, rCLS, SEL, TL])
    #
    # import json
    #
    # lexical_mapping = LexicalMap()
    # train_data = read_cnndm_file(args.raw_data, args, tokenizer, lexical_mapping)
    # torch.save(train_data, args.bert_data)

    # with open('wikihow_data/json_data/train_graph.json','r',encoding='utf-8') as f:
    #     datas = json.load(f)
    #     for data in datas:
    #         print(data['nodes'][0])
    #         if data['nodes'][0].lower()=='how to make a file downloadable from your website5':
    #             print()
# {'sents': ['how to make a file downloadable from your website5', 'if you used the godaddy site builder, log into the godaddy website and open your website in the editor.', 'you can turn any object on your site into a link, as well as any text from your text boxes.', 'if you want to create a download button, click the "button" option from the left menu to insert one.', 'if you have an object selected, click the settings button to open the menu.', 'if you have text selected, click the "link" button in the text formatting tools, which looks like a chainlink.', 'this will allow you to select the file you want to upload to your website.', 'files are limited to 30 mb in size.', 'you cannot upload html, php, exe, dll and several other potentially dangerous types of files.', "you'll see a checkmark next to the file in the window when it has finished uploading.", 'clicking "save" will apply the file to the object or text link you created.', 'this will make your new link live, and your visitors will be able to download the linked file.'], 'depth': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'relation': {0: {0: [{'edge': [], 'length': 0}], 1: [{'edge': ['q_senario'], 'length': 1}], 2: [{'edge': ['q_senario'], 'length': 1}], 3: [{'edge': ['q_senario'], 'length': 1}], 4: [{'edge': ['q_senario'], 'length': 1}], 5: [{'edge': ['q_senario'], 'length': 1}], 6: [{'edge': ['q_senario'], 'length': 1}], 7: [{'edge': ['q_senario'], 'length': 1}], 8: [{'edge': ['q_senario'], 'length': 1}], 9: [{'edge': ['q_senario'], 'length': 1}], 10: [{'edge': ['q_senario'], 'length': 1}], 11: [{'edge': ['q_senario'], 'length': 1}]}, 1: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': [], 'length': 0}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 2: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': [], 'length': 0}], 3: [{'edge': ['samedoc'], 'length': 1}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 3: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['samedoc'], 'length': 1}], 3: [{'edge': [], 'length': 0}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 4: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': [], 'length': 0}], 5: [{'edge': ['samedoc'], 'length': 1}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 5: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['samedoc'], 'length': 1}], 5: [{'edge': [], 'length': 0}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 6: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': [], 'length': 0}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 7: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': [], 'length': 0}], 8: [{'edge': ['samedoc'], 'length': 1}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 8: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['samedoc'], 'length': 1}], 8: [{'edge': [], 'length': 0}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 9: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': [], 'length': 0}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 10: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': [], 'length': 0}], 11: [{'edge': ['senario_q', 'q_senario'], 'length': 2}]}, 11: {0: [{'edge': ['senario_q'], 'length': 1}], 1: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 2: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 3: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 4: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 5: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 6: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 7: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 8: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 9: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 10: [{'edge': ['senario_q', 'q_senario'], 'length': 2}], 11: [{'edge': [], 'length': 0}]}}, 'token': [['open your site in the godaddy site editor.', 'select the object or text that you want to turn into a link.', 'create a link from your selected object or text.', 'click the red arrow below "link (url)" and select "upload.",', 'click the "browse" button and find the file you want to upload.', 'click "insert" once the file has uploaded.', 'click "save" to create the link.', 'click "publish" to save the changes to your site.']], 'tgt': [['open your site in the godaddy site editor.', 'select the object or text that you want to turn into a link.', 'create a link from your selected object or text.', 'click the red arrow below "link (url)" and select "upload.",', 'click the "browse" button and find the file you want to upload.', 'click "insert" once the file has uploaded.', 'click "save" to create the link.', 'click "publish" to save the changes to your site.']], 'raw_tgt': [['open your site in the godaddy site editor.', 'select the object or text that you want to turn into a link.', 'create a link from your selected object or text.', 'click the red arrow below "link (url)" and select "upload.",', 'click the "browse" button and find the file you want to upload.', 'click "insert" once the file has uploaded.', 'click "save" to create the link.', 'click "publish" to save the changes to your site.']], 'copy': [[-1, -1, -1, -1, -1, -1, -1, -1]], 'cp_seq': ['how to make a file downloadable from your website5', 'if you used the godaddy site builder, log into the godaddy website and open your website in the editor.', 'you can turn any object on your site into a link, as well as any text from your text boxes.', 'if you want to create a download button, click the "button" option from the left menu to insert one.', 'if you have an object selected, click the settings button to open the menu.', 'if you have text selected, click the "link" button in the text formatting tools, which looks like a chainlink.', 'this will allow you to select the file you want to upload to your website.', 'files are limited to 30 mb in size.', 'you cannot upload html, php, exe, dll and several other potentially dangerous types of files.', "you'll see a checkmark next to the file in the window when it has finished uploading.", 'clicking "save" will apply the file to the object or text link you created.', 'this will make your new link live, and your visitors will be able to download the linked file.'], 'token2idx': {'if you want to create a download button, click the "button" option from the left menu to insert one.': 30522, 'if you have text selected, click the "link" button in the text formatting tools, which looks like a chainlink.': 30523, 'if you used the godaddy site builder, log into the godaddy website and open your website in the editor.': 30524, 'this will allow you to select the file you want to upload to your website.': 30525, 'how to make a file downloadable from your website5': 30526, 'you cannot upload html, php, exe, dll and several other potentially dangerous types of files.': 30527, 'if you have an object selected, click the settings button to open the menu.': 30528, "you'll see a checkmark next to the file in the window when it has finished uploading.": 30529, 'clicking "save" will apply the file to the object or text link you created.': 30530, 'you can turn any object on your site into a link, as well as any text from your text boxes.': 30531, 'this will make your new link live, and your visitors will be able to download the linked file.': 30532, 'files are limited to 30 mb in size.': 30533}, 'idx2token': {30522: 'if you want to create a download button, click the "button" option from the left menu to insert one.', 30523: 'if you have text selected, click the "link" button in the text formatting tools, which looks like a chainlink.', 30524: 'if you used the godaddy site builder, log into the godaddy website and open your website in the editor.', 30525: 'this will allow you to select the file you want to upload to your website.', 30526: 'how to make a file downloadable from your website5', 30527: 'you cannot upload html, php, exe, dll and several other potentially dangerous types of files.', 30528: 'if you have an object selected, click the settings button to open the menu.', 30529: "you'll see a checkmark next to the file in the window when it has finished uploading.", 30530: 'clicking "save" will apply the file to the object or text link you created.', 30531: 'you can turn any object on your site into a link, as well as any text from your text boxes.', 30532: 'this will make your new link live, and your visitors will be able to download the linked file.', 30533: 'files are limited to 30 mb in size.'}, 'idx2tokenid': {30522: [2065, 2017, 2215, 2000, 3443, 1037, 8816, 6462, 1010, 11562, 1996, 1000, 6462, 1000, 5724, 2013, 1996, 2187, 12183, 2000, 19274, 2028, 1012], 30523: [2065, 2017, 2031, 3793, 3479, 1010, 11562, 1996, 1000, 4957, 1000, 6462, 1999, 1996, 3793, 4289, 3436, 5906, 1010, 2029, 3504, 2066, 1037, 4677, 13767, 1012], 30524: [2065, 2017, 2109, 1996, 2643, 4215, 5149, 2609, 12508, 1010, 8833, 2046, 1996, 2643, 4215, 5149, 4037, 1998, 2330, 2115, 4037, 1999, 1996, 3559, 1012], 30525: [2023, 2097, 3499, 2017, 2000, 7276, 1996, 5371, 2017, 2215, 2000, 2039, 11066, 2000, 2115, 4037, 1012], 30526: [2129, 2000, 2191, 1037, 5371, 26720, 2013, 2115, 4037, 2629], 30527: [2017, 3685, 2039, 11066, 16129, 1010, 25718, 1010, 4654, 2063, 1010, 21469, 2140, 1998, 2195, 2060, 9280, 4795, 4127, 1997, 6764, 1012], 30528: [2065, 2017, 2031, 2019, 4874, 3479, 1010, 11562, 1996, 10906, 6462, 2000, 2330, 1996, 12183, 1012], 30529: [2017, 1005, 2222, 2156, 1037, 4638, 10665, 2279, 2000, 1996, 5371, 1999, 1996, 3332, 2043, 2009, 2038, 2736, 2039, 18570, 1012], 30530: [22042, 1000, 3828, 1000, 2097, 6611, 1996, 5371, 2000, 1996, 4874, 2030, 3793, 4957, 2017, 2580, 1012], 30531: [2017, 2064, 2735, 2151, 4874, 2006, 2115, 2609, 2046, 1037, 4957, 1010, 2004, 2092, 2004, 2151, 3793, 2013, 2115, 3793, 8378, 1012], 30532: [2023, 2097, 2191, 2115, 2047, 4957, 2444, 1010, 1998, 2115, 5731, 2097, 2022, 2583, 2000, 8816, 1996, 5799, 5371, 1012], 30533: [6764, 2024, 3132, 2000, 2382, 16914, 1999, 2946, 1012]}, 'tgt_subtoken': ['[unused1]', 'open', 'your', 'site', 'in', 'the', 'god', '##ad', '##dy', 'site', 'editor', '.', '[unused3]', 'select', 'the', 'object', 'or', 'text', 'that', 'you', 'want', 'to', 'turn', 'into', 'a', 'link', '.', '[unused3]', 'create', 'a', 'link', 'from', 'your', 'selected', 'object', 'or', 'text', '.', '[unused3]', 'click', 'the', 'red', 'arrow', 'below', '"', 'link', '(', 'ur', '##l', ')', '"', 'and', 'select', '"', 'up', '##load', '.', '"', ',', '[unused3]', 'click', 'the', '"', 'brows', '##e', '"', 'button', 'and', 'find', 'the', 'file', 'you', 'want', 'to', 'up', '##load', '.', '[unused3]', 'click', '"', 'insert', '"', 'once', 'the', 'file', 'has', 'uploaded', '.', '[unused3]', 'click', '"', 'save', '"', 'to', 'create', 'the', 'link', '.', '[unused3]', 'click', '"', 'publish', '"', 'to', 'save', 'the', 'changes', 'to', 'your', 'site', '.', '[unused2]'], 'sents_pre': [{'tokens': '[CLS] how to make a file downloadable from your website ##5 [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] if you used the god ##ad ##dy site builder , log into the god ##ad ##dy website and open your website in the editor . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] you can turn any object on your site into a link , as well as any text from your text boxes . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] if you want to create a download button , click the " button " option from the left menu to insert one . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] if you have an object selected , click the settings button to open the menu . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] if you have text selected , click the " link " button in the text format ##ting tools , which looks like a chain ##link . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] this will allow you to select the file you want to up ##load to your website . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] files are limited to 30 mb in size . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] you cannot up ##load html , php , ex ##e , dl ##l and several other potentially dangerous types of files . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': "[CLS] you ' ll see a check ##mark next to the file in the window when it has finished up ##loading . [SEP]", 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] clicking " save " will apply the file to the object or text link you created . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {'tokens': '[CLS] this will make your new link live , and your visitors will be able to download the linked file . [SEP]', 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}], 'tgt_subtokens_out': ['[unused1]', 'open', 'your', 'site', 'in', 'the', 'god', '##ad', '##dy', 'site', 'editor', '.', '[unused3]', 'select', 'the', 'object', 'or', 'text', 'that', 'you', 'want', 'to', 'turn', 'into', 'a', 'link', '.', '[unused3]', 'create', 'a', 'link', 'from', 'your', 'selected', 'object', 'or', 'text', '.', '[unused3]', 'click', 'the', 'red', 'arrow', 'below', '"', 'link', '(', 'ur', '##l', ')', '"', 'and', 'select', '"', 'up', '##load', '.', '"', ',', '[unused3]', 'click', 'the', '"', 'brows', '##e', '"', 'button', 'and', 'find', 'the', 'file', 'you', 'want', 'to', 'up', '##load', '.', '[unused3]', 'click', '"', 'insert', '"', 'once', 'the', 'file', 'has', 'uploaded', '.', '[unused3]', 'click', '"', 'save', '"', 'to', 'create', 'the', 'link', '.', '[unused3]', 'click', '"', 'publish', '"', 'to', 'save', 'the', 'changes', 'to', 'your', 'site', '.', '[unused2]'], 'raw_tgt_subtokens_str': ['[unused1]', 'open', 'your', 'site', 'in', 'the', 'god', '##ad', '##dy', 'site', 'editor', '.', '[unused3]', 'select', 'the', 'object', 'or', 'text', 'that', 'you', 'want', 'to', 'turn', 'into', 'a', 'link', '.', '[unused3]', 'create', 'a', 'link', 'from', 'your', 'selected', 'object', 'or', 'text', '.', '[unused3]', 'click', 'the', 'red', 'arrow', 'below', '"', 'link', '(', 'ur', '##l', ')', '"', 'and', 'select', '"', 'up', '##load', '.', '"', ',', '[unused3]', 'click', 'the', '"', 'brows', '##e', '"', 'button', 'and', 'find', 'the', 'file', 'you', 'want', 'to', 'up', '##load', '.', '[unused3]', 'click', '"', 'insert', '"', 'once', 'the', 'file', 'has', 'uploaded', '.', '[unused3]', 'click', '"', 'save', '"', 'to', 'create', 'the', 'link', '.', '[unused3]', 'click', '"', 'publish', '"', 'to', 'save', 'the', 'changes', 'to', 'your', 'site', '.', '[unused2]'], 'src_tokens': ['[CLS]', 'how', 'to', 'make', 'a', 'file', 'downloadable', 'from', 'your', 'website', '##5', '[SEP]', '[CLS]', 'if', 'you', 'used', 'the', 'god', '##ad', '##dy', 'site', 'builder', ',', 'log', 'into', 'the', 'god', '##ad', '##dy', 'website', 'and', 'open', 'your', 'website', 'in', 'the', 'editor', '.', '[SEP]', '[CLS]', 'you', 'can', 'turn', 'any', 'object', 'on', 'your', 'site', 'into', 'a', 'link', ',', 'as', 'well', 'as', 'any', 'text', 'from', 'your', 'text', 'boxes', '.', '[SEP]', '[CLS]', 'if', 'you', 'want', 'to', 'create', 'a', 'download', 'button', ',', 'click', 'the', '"', 'button', '"', 'option', 'from', 'the', 'left', 'menu', 'to', 'insert', 'one', '.', '[SEP]', '[CLS]', 'if', 'you', 'have', 'an', 'object', 'selected', ',', 'click', 'the', 'settings', 'button', 'to', 'open', 'the', 'menu', '.', '[SEP]', '[CLS]', 'if', 'you', 'have', 'text', 'selected', ',', 'click', 'the', '"', 'link', '"', 'button', 'in', 'the', 'text', 'format', '##ting', 'tools', ',', 'which', 'looks', 'like', 'a', 'chain', '##link', '.', '[SEP]', '[CLS]', 'this', 'will', 'allow', 'you', 'to', 'select', 'the', 'file', 'you', 'want', 'to', 'up', '##load', 'to', 'your', 'website', '.', '[SEP]', '[CLS]', 'files', 'are', 'limited', 'to', '30', 'mb', 'in', 'size', '.', '[SEP]', '[CLS]', 'you', 'cannot', 'up', '##load', 'html', ',', 'php', ',', 'ex', '##e', ',', 'dl', '##l', 'and', 'several', 'other', 'potentially', 'dangerous', 'types', 'of', 'files', '.', '[SEP]', '[CLS]', 'you', "'", 'll', 'see', 'a', 'check', '##mark', 'next', 'to', 'the', 'file', 'in', 'the', 'window', 'when', 'it', 'has', 'finished', 'up', '##loading', '.', '[SEP]', '[CLS]', 'clicking', '"', 'save', '"', 'will', 'apply', 'the', 'file', 'to', 'the', 'object', 'or', 'text', 'link', 'you', 'created', '.', '[SEP]', '[CLS]', 'this', 'will', 'make', 'your', 'new', 'link', 'live', ',', 'and', 'your', 'visitors', 'will', 'be', 'able', 'to', 'download', 'the', 'linked', 'file', '.', '[SEP]', '[SEP]'], 'segments': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'cls_ids': [0, 12, 39, 63, 88, 106, 134, 153, 164, 188, 211, 230]}

        # print()