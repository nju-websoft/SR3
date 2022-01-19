# encoding=utf8
import re
import random
import networkx as nx
import evaluate
import numpy as np
import copy
class sentGraph(object):

    def comfirmConnect(self,relations, nodes,node_types):
        topk_idx=[]
        contain_nodes=[]
        for idx,(node,node_type,relation) in enumerate(zip(nodes,node_types,relations)):
            # if node not in nodes:
            if idx not in topk_idx and node not in contain_nodes:
                if node_type=='question' or (node_type!='summary' and node_type!='geonames'):
                    topk_idx.append(idx)
                    contain_nodes.append(node)
                else:
                    couldconnect=False
                    for idx2 in topk_idx:
                        if relation[idx2]!=None:
                            couldconnect=True
                            break
                    if not couldconnect:
                        for idx2 in range(idx+1,len(nodes)):
                            if relation[idx2]=='sameEntity' and (node_types[idx2]!='summary' and node_types[idx2]!='geonames') and idx2 not in topk_idx and nodes[idx2] not in contain_nodes:
                                topk_idx.append(idx2)
                                contain_nodes.append(nodes[idx2])
                                break
                    topk_idx.append(idx)
                    contain_nodes.append(node)
        # print(topk_idx)
        return topk_idx

    def getMaxNum(self,nodes,args):
        size = 0
        idx=0
        for sent in nodes:
            size+=len(self.tokenizer.tokenizer.tokenize(sent))+2
            if size>args.max_pos:
                break
            idx+=1
        return  idx

    def getTopK(self,nodes_raworder,nodes,args):
        maxNum = self.getMaxNum(nodes,args)
        nodes=nodes[:maxNum]
        nodes_raworder=nodes_raworder[:maxNum]
        scores_sort = list(np.argsort(np.array(nodes_raworder)))
        return scores_sort

    def __init__(self, relations, nodes, tgt,node_types,args,tokenizer,nodes_raworder=None,nodes_tag=None):
        # if args.split_qm:
        if args.max_node_size<0:
            max_node_size = len(nodes)
        else:
            max_node_size = args.max_node_size
        # print(nodes)
        self.tokenizer = tokenizer
        if args.dataset == 'geo':
            if args.comfirm_connect:
                topk_idx = self.comfirmConnect(relations, nodes, node_types)[:max_node_size]
            else:
                topk_idx = [i for i in range(max_node_size) if i < len(nodes)]
            if args.recovery_order:
                nodes_raworder = [nodes_raworder[idx] for idx in topk_idx]
                scores_sort = list(np.argsort(np.array(nodes_raworder)))
                topk_idx = [topk_idx[i] for i in scores_sort]
        else:
            if args.recovery_order:
                topk_idx = self.getTopK(nodes_raworder, nodes, args)
            else:
                topk_idx = [i for i in range(max_node_size) if i < len(nodes)]
        # if args.comfirm_connect and args.dataset=='geo':
        #     topk_idx=self.comfirmConnect(relations, nodes,node_types)[:max_node_size]
        #     if args.recovery_order:
        #         nodes_raworder = [nodes_raworder[idx].lower() for idx in topk_idx]
        #         scores_sort = list(np.argsort(np.array(nodes_raworder)))
        #         topk_idx= [topk_idx[i] for i in scores_sort]
        # elif args.recovery_order:
        #     topk_idx = self.getTopK( nodes_raworder, nodes,args)
        # else:
        #     topk_idx = [i for i in range(max_node_size) if i < len(nodes)]
        if args.dataset=='geo':
            nodes = [nodes[idx].replace(" ", '').replace('。', '，').lower() for idx in
                     topk_idx]
        else:
            nodes = [nodes[idx].lower() for idx in topk_idx]
        # nodes = [nodes[idx].replace(" ",'') if idx ==0 else nodes[idx].replace(" ",'').replace('。','，') for idx in topk_idx]
        # node_types = [node_types[idx] for idx in topk_idx]
        if relations!=None:
            relations = [[relations[idx][idx2] for idx2 in topk_idx] for idx in topk_idx]
        if nodes_tag!=None:
            self.nodes_tag = [1]+[nodes_tag[idx] for idx in topk_idx][1:]
        # nodes=nodes[:max_node_size]
        # node_types=node_types[:max_node_size]
        # relations=[rel[:max_node_size] for rel in relations[:max_node_size]]
        # transform graph from original conll format into our own data structure
        self.graph = nx.DiGraph()
        self.name2concept = nodes
        # self.node_types=node_types
        self.args = args
        self.root = None
        for i, x in enumerate(nodes):
            if i == 0:
                assert self.root is None
                self.root = i
            self.graph.add_node(i)

        if relations!=None:
            for src,dess in enumerate(relations):
                for des,rel in enumerate(dess):
                    if rel:
                        self._add_edge(rel, src, des)

        self.target = tgt
        self.raw_tgt = copy.deepcopy(tgt)


    def __len__(self):
        return len(self.name2concept) ** 2 + len(self.target)

    def _add_edge(self, rel, src, des):
        self.graph.add_node(src)
        self.graph.add_node(des)
        self.graph.add_edge(src, des, label=rel)
        # self.graph.add_edge(des, src, label=rel + '_r_')

    # 尝试非连通图的bfs
    def bfs(self):
        g = self.graph
        queue = [self.root]
        depths = [0]
        visited = set(queue)
        step = 0
        while step < len(queue):
            u = queue[step]
            depth = depths[step]
            step += 1
            for v in g.neighbors(u):
                if v not in visited:
                    queue.append(v)
                    depths.append(depth + 1)
                    visited.add(v)
        is_connected = (len(queue) == g.number_of_nodes())

        # nodes = [i for i in range(len(self.name2concept))]
        # depths = [depths[queue.index(i)] for i in nodes]
        # queue = nodes
        return queue, depths, is_connected

    def collect_sents_and_relations_confirm_connect(self):
        g = self.graph
        nodes, depths, is_connected = self.bfs()
        sents = [self.name2concept[n] for n in nodes]
        self.nodes_tag = [self.nodes_tag[n] for n in nodes]
        relations = dict()
        for i, src in enumerate(nodes):
            relations[i] = dict()
            paths = nx.single_source_shortest_path(g, src)
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                assert tgt in paths
                path = paths[tgt]
                info = dict()
                # info['node'] = path[1:-1]
                info['edge'] = [g[path[i]][path[i + 1]]['label'] for i in range(len(path) - 1)]
                info['length'] = len(info['edge'])
                relations[i][j].append(info)

        ## TODO, we just use the sequential order
        depths = nodes
        self.name2concept = sents
        return sents, depths, relations, is_connected

    def collect_sents_and_relations(self):
        g = self.graph
        nodes = [i for i in range(len(self.name2concept))]
        self.nodes_tag = [self.nodes_tag[n] for n in nodes]
        sents = self.name2concept
        relations = dict()
        for i, src in enumerate(nodes):
            relations[i] = dict()
            paths = nx.single_source_shortest_path(g, src)
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                if tgt in paths:
                    path = paths[tgt]
                    info = dict()
                    # info['node'] = path[1:-1]
                    info['edge'] = [g[path[i]][path[i + 1]]['label'] for i in range(len(path) - 1)]
                    info['length'] = len(info['edge'])
                else:
                    info = dict()
                    # info['node'] = path[1:-1]
                    info['edge'] = ['not_connect']
                    info['length'] = len(info['edge'])
                relations[i][j].append(info)

        ## TODO, we just use the sequential order
        depths = nodes
        self.name2concept = sents
        return sents, depths, relations

    def addTgtCopy(self):
        self.copy=[]
        if self.args.dataset!="cnndm":
            copysents = self.name2concept[1:]
        else:
            copysents = self.name2concept
        for tgt in self.target:
            copy=[-1 for i in range(len(tgt))]
            copy_score = [-1 for i in range(len(tgt))]

            for idx, node in enumerate(copysents):
                maxscore = -1
                max_idx = -1
                for s_id, sent in enumerate(tgt):
                    if self.args.use_rouge_f:
                        metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f'))
                    else:
                        metrics = (('rouge-1', 'r'),('rouge-1', 'p'),('rouge-1', 'f'),
                                   ('rouge-2', 'r'),('rouge-2', 'p'),('rouge-2', 'f'),
                                   ('rouge-l', 'r'),('rouge-l', 'p'),('rouge-l', 'f'))
                    if self.args.dataset=='geo':
                        score=evaluate.evalAnswer(sent,node,metrics=metrics)
                    else:
                        score = evaluate.evalAnswer(sent, node,language='en',metrics=metrics,max_len=500)
                    if self.args.use_rouge_f:
                        score = (score[0]+score[1])/2
                    else:
                        # print(score)
                        score = (min(score[0],score[1])+min(score[3],score[4]))/2
                        # print(score)
                    if score>maxscore:
                        maxscore=score
                        max_idx=s_id
                if maxscore>self.args.min_copy_rouge and maxscore>copy_score[max_idx]:
                    if self.args.dataset!="cnndm":
                        copy_idx = idx+1
                    else:
                        copy_idx = idx
                    copy[max_idx]=copy_idx
                    copy_score[max_idx] = maxscore
                    print('$copy$:',tgt[max_idx])
                    print(self.name2concept[copy_idx],maxscore)
                    # print()

            self.copy.append(copy)
