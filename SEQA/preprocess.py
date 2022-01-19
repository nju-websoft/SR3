from os.path import join as pjoin
import json
import argparse
from prepro import tokenizer,tokenizer_gen,tokenized_gen_pred,tokenizer_copy
import torch
def format_to_bert(args):
    datasets=args.dataset
    bert_tokenizer=tokenizer.BertData(args)
    for dataset in datasets:
        readfile=pjoin(args.raw_path,dataset+'.json')
        writefile=pjoin(args.pre_path,dataset+'.bert.pt')
        rawdatas=json.load(open(readfile,'r',encoding='utf-8'))
        bertdatas=[]
        for data in rawdatas:
            positive_items, negative_items,tgt_subtoken_idxs,src_tokens_idxs=bert_tokenizer.preprocess(data,dataset)
            bertdata={"qid":data["id"],
                      "src_tokens_idxs":src_tokens_idxs,
                      "raw_src":data['src'],
                      "positive_items":positive_items,
                      "negative_items":negative_items,
                      "tgt_token_idxs":tgt_subtoken_idxs,
                      "raw_tgt":data['tgt'][0]}
            bertdatas.append(bertdata)
        torch.save(bertdatas, writefile)
        # json.dump(bertdatas,open(writefile,'w',encoding='utf-8'),ensure_ascii=False)

def format_to_bert_gen(args):
    datasets=args.dataset
    bert_tokenizer=tokenizer_gen.BertData(args)
    for dataset in datasets:
        readfile=pjoin(args.raw_path,dataset+'.json')
        writefile=pjoin(args.pre_path,dataset+'.bert.pt')
        rawdatas=json.load(open(readfile,'r',encoding='utf-8'))
        bertdatas=[]
        for data in rawdatas:
            if not args.tag_ext:
                tgt_subtoken_idxs, src_tokens_idxs, segments, types,src_tokens=bert_tokenizer.preprocess(data)
                bertdata={"qid":data["id"],
                          "src_tokens_idxs":src_tokens_idxs,
                          "src_tokens":src_tokens,
                          "raw_src":data['src'],
                          "types":types,
                          "segments":segments,
                          "tgt_token_idxs":tgt_subtoken_idxs,
                          "raw_tgt":data['tgt'][0]}
                bertdatas.append(bertdata)
            else:
                if len(data['goldsent'])>0:
                    tgt_subtoken_idxs, src_tokens_idxs, segments, types, src_tokens,ext_subtoken_idxs,ext_split_ids,ext_tokens = bert_tokenizer.preprocess(data)
                    bertdata = {"qid": data["id"],
                                "src_tokens_idxs": src_tokens_idxs,
                                "src_tokens": src_tokens,
                                "raw_src": data['src'],
                                "types": types,
                                "segments": segments,
                                "tgt_token_idxs": tgt_subtoken_idxs,
                                "raw_tgt": data['tgt'][0],
                                "ext_subtoken_idxs":ext_subtoken_idxs,
                                "ext_split_ids":ext_split_ids,
                                "ext_tokens":ext_tokens}
                    bertdatas.append(bertdata)
        torch.save(bertdatas, writefile)
        # json.dump(bertdatas, open(writefile, 'w', encoding='utf-8'), ensure_ascii=False)

def format_to_bert_copy(args):
    datasets=args.dataset
    bert_tokenizer=tokenizer_copy.BertData(args)
    for dataset in datasets:
        readfile=pjoin(args.raw_path,dataset+'.json')
        writefile=pjoin(args.pre_path,dataset+'.bert.pt')
        rawdatas=json.load(open(readfile,'r',encoding='utf-8'))
        bertdatas=[]
        for data in rawdatas:
            tgt_subtoken_idxs,tgt_subtoken, src_tokens_idxs, segments, types, src_tokens, ext_subtoken_idxs, ext_split_ids, ext_subtoken, ext_sent_len, ext_copy=bert_tokenizer.preprocess(data)
            bertdata={"qid":data["id"],
                      "src_tokens_idxs":src_tokens_idxs,
                      "src_tokens":src_tokens,
                      "raw_src":data['src'],
                      "segments":segments,
                      "tgt_token_idxs":tgt_subtoken_idxs,
                      "tgt_subtoken":tgt_subtoken,
                      "ext_subtoken_idxs":ext_subtoken_idxs,
                      "ext_tokens":ext_subtoken,
                      "ext_split_ids":ext_split_ids,
                      "ext_copy":ext_copy,
                      "ext_sent_len":ext_sent_len,
                      "raw_tgt":data['tgt'][0]}
            bertdatas.append(bertdata)
        torch.save(bertdatas, writefile)

def format_to_bert_gen_pred(args):
    datasets=args.dataset
    bert_tokenizer=tokenized_gen_pred.BertData(args)
    for dataset in datasets:
        readfile=pjoin(args.raw_path,dataset+'.json')
        writefile=pjoin(args.pre_path,dataset+'.bert.pt')
        rawdatas=json.load(open(readfile,'r',encoding='utf-8'))
        bertdatas=[]
        for data in rawdatas:
            tgt_subtoken_idxs, src_tokens_idxs, segments,src_tokens=bert_tokenizer.preprocess(data,dataset)
            bertdata={"qid":data["id"],
                      "src_tokens_idxs":src_tokens_idxs,
                      "src_tokens":src_tokens,
                      "raw_src":data['src'],
                      "segments":segments,
                      "tgt_token_idxs":tgt_subtoken_idxs,
                      "raw_tgt":data['tgt'][0]}
            bertdatas.append(bertdata)
        torch.save(bertdatas, writefile)

import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='classify', type=str,choices=['gen','classify','gen_pred','copy'])
    parser.add_argument("-raw_path", default='data/raw_data', type=str)
    parser.add_argument("-pre_path", default='data/pre_data', type=str)
    parser.add_argument("-dataset",default=['train','test'],type=list)
    parser.add_argument("-bertname",default='bert-base-chinese',type=str)
    parser.add_argument("-maxdoclen",default=512,type=int)
    parser.add_argument("-max_tgt_ntokens", default=128, type=int)
    parser.add_argument("-max_positive", default=10, type=int)
    parser.add_argument("-max_negative", default=20, type=int)

    parser.add_argument("-tag_ext", default=False, type=bool)

    args = parser.parse_args()
    # args.task='gen_pred'
    # args.raw_path='gen_data_pred/raw_data'
    # args.pre_path='gen_data_pred/pred_data'

    # args.task='gen'
    # args.raw_path='gen_data/raw_ext_data'
    # args.pre_path='gen_data/pre_ext_data'
    # args.tag_ext=True

    # args.task = 'copy'
    # args.raw_path = 'copy_data/raw_data'
    # args.pre_path = 'copy_data/pre_data'
    #
    # if args.task=="classify":
    #     format_to_bert(args)
    # if args.task == "gen":
    #     format_to_bert_gen(args)
    # if args.task == "gen_pred":
    #     format_to_bert_gen_pred(args)
    # if args.task == "copy":
    #     format_to_bert_copy(args)

    datas = torch.load('copy_data/pre_data/test.bert.pt')
    # for data in datas:
    #     print(data['src_tokens_idxs'][-1])
    #     print(data['src_tokens'][-1])
    print()
    # with open('data/train.bert.pt.json','r',encoding='utf-8') as f:
    #     datas=json.load(f)
    #     positive_lens=[]
    #     negative_lens=[]
    #     for data in datas:
    #         positive_len=len(data["positive_items"])
    #         negative_len=len(data['negative_items'])
    #         positive_lens.append(positive_len)
    #         negative_lens.append(negative_len)
    #
    #     print(np.max(positive_lens),np.min(positive_lens),np.mean(positive_lens))
    #     print(np.max(negative_lens), np.min(negative_lens), np.mean(negative_lens))