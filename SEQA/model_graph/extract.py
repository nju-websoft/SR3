#!/usr/bin/env python
# coding: utf-8
from collections import Counter
from model_graph.sentGraph import sentGraph
import torch
import json
class IO:

    def __init__(self):
        pass

    @staticmethod
    def read(file_path,args,tokenizer):
        with open(file_path,encoding='utf-8') as f:
            datas=json.load(f)
            for data in datas:
                if args.dataset=='geo':
                    relations, nodes, tgt, node_types,nodes_raworder,node_labels=data['relations'],data['nodes'],data['tgt'],data['node_types'],eval(data['nodes_raworder']),data['node_labels']
                    yield sentGraph(relations, nodes, tgt,node_types,args,tokenizer,nodes_raworder,node_labels)
                elif 'relations' not in data:
                    relations, nodes, tgt, node_types = None, data['article'], [data['answer']], None
                    if "node_labels" in data:
                        node_labels = data['node_labels']
                    else:
                        node_labels = data['article_tags']
                    temp = []
                    labels = []
                    if "question" in data:
                        temp.extend(data['question'])
                        labels.append(1)
                    for node,l in zip(nodes,node_labels):
                        temp.extend(node)
                        labels.extend(l)
                    nodes=temp
                    nodes_raworder = [i for i in range(len(nodes))]
                    node_labels=labels
                    yield sentGraph(relations, nodes, tgt, None, args, tokenizer, nodes_raworder,node_labels)
                else:
                    relations, nodes, tgt,nodes_raworder,node_labels = data['relations'], data['nodes'], data['tgt'],eval(data['nodes_raworder']),data['node_labels']
                    yield sentGraph(relations, nodes, tgt, None, args,tokenizer,nodes_raworder,node_labels)

class LexicalMap(object):

    # build our lexical mapping (from concept to token/lemma), useful for copy mechanism.
    def __init__(self):
        pass

    #cp_seq, token2idx, idx2token = lex_map.get(concept, vocabs['predictable_token'])
    @staticmethod
    def get(concept, tokenizer,args):
        cp_seq = []
        for conc in concept:
            cp_seq.append(conc)

        new_tokens = set(cp_seq)
        token2idx, idx2token = dict(), dict()
        idx2tokenid = dict()
        nxt = tokenizer.tokenizer.vocab_size
        for x in new_tokens:
            # x=x
            token2idx[x] = nxt
            idx2token[nxt] = x
            if args.dataset=='geo':
                idx2tokenid[nxt] = [tokenizer.tokenizer.convert_tokens_to_ids(t) for t in tokenizer.tokenizer.tokenize(' '.join([word for word in x]))]
            else:
                idx2tokenid[nxt] = [tokenizer.tokenizer.convert_tokens_to_ids(t) for t in
                                    tokenizer.tokenizer.tokenize(x)]
            nxt += 1
        return cp_seq, token2idx, idx2token,idx2tokenid




def read_file(filename,args,tokenizer,lex_map):
    # read prepared file
    def work(graph,tokenizer,lex_map):
        if args.comfirm_connect:
            sent, depth, relation, ok = graph.collect_sents_and_relations_confirm_connect()
            if not ok:
                print()
            assert ok, "not connected"
        else:
            sent, depth, relation = graph.collect_sents_and_relations()
        if args.copy_decoder and args.split_qm:
            graph.addTgtCopy()
        else:
            graph.copy = [[-1 for i in range(len(tgt))] for tgt in graph.target]
    # if args.split_qm:
        tok = graph.target
        cp_seq, token2idx, idx2token,idx2tokenid = lex_map.get(sent, tokenizer,args)
        # from sentGraph import max_node_size
        # if len(token2idx.keys())!=max_node_size:
        #     print(sent)
        item = {'sents': sent,
                'depth': depth,
                'node_labels':graph.nodes_tag,
                'relation': relation,
                'token': tok,
                'tgt':graph.target,
                'raw_tgt':graph.raw_tgt,
                'copy':graph.copy,
                'cp_seq': cp_seq,
                'token2idx': token2idx,
                'idx2token': idx2token,
                'idx2tokenid':idx2tokenid
        }
        # else:
        #     item = {
        #             'sents': sent,
        #             'token': graph.target,
        #             'tgt': graph.target,
        #             'raw_tgt': graph.raw_tgt,
        #             'copy': graph.copy,
        #             }
        tgt_subtoken, sents_pre, tgt_subtokens_out, raw_tgt_subtokens_str, src_tokens, segments,cls_ids = tokenizer.preprocess(
            item)
        item['tgt_subtoken']=tgt_subtoken
        item['sents_pre'] = sents_pre
        item['tgt_subtokens_out'] = tgt_subtokens_out
        item['raw_tgt_subtokens_str'] = raw_tgt_subtokens_str
        item['src_tokens'] = src_tokens
        item['segments']=segments
        item['cls_ids'] = cls_ids
        if len(item['tgt_subtoken']) != len(item['tgt_subtokens_out']):
            print(item)
        # assert len(item['tgt_subtoken']) == len(item['tgt_subtokens_out'])
        return item

    data = []
    for graph in IO.read(filename,args,tokenizer):
        graph = work(graph,tokenizer,lex_map)
        # if len(graph['tgt_subtoken']) != len(graph['tgt_subtokens_out']):
        if len(graph['tgt_subtoken']) == len(graph['tgt_subtokens_out']):
            data.append(graph)
        else:
            print(graph)
    print ('read from %s, %d instances'%(filename, len(data)))
    return data


def read_cnndm_file(filename,args,tokenizer,lex_map):
    # read prepared file
    def work(graph,tokenizer,lex_map):
        sent = graph.name2concept
        if args.split_qm:
            cp_seq, token2idx, idx2token,idx2tokenid = lex_map.get(sent, tokenizer,args)
            item = {'sents': sent,
                    'tgt':graph.target,
                    'raw_tgt':graph.raw_tgt,
                    'copy':graph.copy,
                    'cp_seq': cp_seq,
                    'token2idx': token2idx,
                    'idx2token': idx2token,
                    'idx2tokenid':idx2tokenid
            }
        else:
            item = {'sents': sent,
                    'tgt': graph.target,
                    'raw_tgt': graph.raw_tgt,
                    }
        if args.copy_decoder and args.split_qm:
            graph.addTgtCopy()
        else:
            graph.copy = [[-1 for i in range(len(tgt))] for tgt in graph.target]
        tgt_subtoken, sents_pre, tgt_subtokens_out, raw_tgt_subtokens_str, src_tokens, segments,cls_ids = tokenizer.preprocess(
            item)
        item['tgt_subtoken']=tgt_subtoken
        # item['sents_pre'] = sents_pre
        item['tgt_subtokens_out'] = tgt_subtokens_out
        item['raw_tgt_subtokens_str'] = raw_tgt_subtokens_str
        item['src_tokens'] = src_tokens
        item['segments']=segments
        item['cls_ids'] = cls_ids
        return item

    data = []
    i=0
    copy_count=0
    for idx,graph in enumerate(IO.read(filename,args,tokenizer)):
        if len(graph.name2concept)>0:
            for c in graph.copy[0]:
                if c!=-1:
                    copy_count+=1
            graph = work(graph,tokenizer,lex_map)
            data.append(graph)
            i+=1
    print(copy_count)
    print ('read from %s, %d instances'%(filename, len(data)))
    return data


def make_vocab(cnt, char_level=False):
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w',encoding='utf-8') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))


