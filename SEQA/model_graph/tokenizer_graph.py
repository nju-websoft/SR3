from pytorch_transformers import BertTokenizer
import random
import copy

class BertData():
    def __init__(self,args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.encoder_name, do_lower_case=True)

        # self.tokenizer = BertTokenizer.from_pretrained(args.encoder_name,cache_dir=args.temp_dir, do_lower_case=True)
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_sent_split = '[unused3]'
        self.sent_entity = '[unused4]'
        self.entity_split = '[unused5]'
        self.doc_entity = '[unused6]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.split_vid = self.tokenizer.vocab[self.tgt_sent_split]
        self.doc_entity_vid = self.tokenizer.vocab[self.doc_entity]

    def preprocess_src(self,src):
        sent_tokens=[self.preprocess_sent(src)]
        sent_tokens = ' {} {} '.format(self.sep_token, self.cls_token).join(sent_tokens)
        sent_tokens = sent_tokens.split()
        tokens = [self.cls_token] + sent_tokens + [self.sep_token]
        return tokens

    def preprocess_sent(self,sent):
        if self.args.dataset == 'geo':
            sent_token = ' '.join(self.tokenizer.tokenize(' '.join([word for word in sent])))
        else:
            sent_token = ' '.join(self.tokenizer.tokenize(sent))
        return sent_token

    def preprocess_test(self,sents):
        sent_tokens=[]
        segments=[]
        for idx,sent in enumerate(sents):
            tokens = self.preprocess_sent(sent)
            sent_tokens.append(tokens)
            if self.args.dataset == "cnndm" and idx%2==0:
                segments .extend( [1] * (len(tokens.split())+2))
            else:
                segments .extend( [0] * (len(tokens.split())+2))
        sent_tokens = ' {} {} '.format(self.sep_token, self.cls_token).join(sent_tokens)
        sent_tokens = sent_tokens.split()
        tokens = [self.cls_token]  + sent_tokens + [self.sep_token]
        return tokens,segments

    def preprocess(self,data):
        sents = data['sents']
        if len(sents)<=0:
            print(data)
        raw_src_tokens = self.preprocess_src(sents[0])
        sents_pre=[]
        sent_token0 = '[CLS] ' + ' '.join(self.tokenizer.tokenize(sents[0])) + ' [SEP]'
        segments0 = [0] * len(sent_token0.split())
        # sents_idx=self.tokenizer.convert_tokens_to_ids(sent_token)
        sents_pre.append({"tokens": sent_token0, "segments": segments0})

        for idx,sent in enumerate(sents[1:]):
            sent_token = '[CLS] '+' '.join(self.tokenizer.tokenize(sent))+' [SEP]'
            # segments = [0] * len(sent_token0.split()) + [1] * len(sent_token.split())
            # src_tokens = sent_token0 +' '+ sent_token
            if self.args.dataset == "cnndm" and idx%2==0:
                segments = [1] * len(sent_token.split())
            else:
                segments = [0] * len(sent_token.split())
            src_tokens = sent_token
            sents_pre.append({"tokens": src_tokens, "segments": segments})

        doc_tokens,segments = self.preprocess_test(sents[1:])
        if self.args.dataset == "cnndm":
            segments = [0] * len(raw_src_tokens) + segments
        else:
            segments = [0] * len(raw_src_tokens) + [1] * len(doc_tokens)
        src_tokens = raw_src_tokens + doc_tokens
        # print(''.join(src_tokens))
        end_token = [src_tokens[-1]]
        end_seg = [segments[-1]]
        src_tokens = src_tokens[:self.args.max_pos - 1] + end_token
        segments = segments[:self.args.max_pos - 1] + end_seg


        tgt=data['tgt'][0]
        if self.args.dataset == 'geo':
            tgt = [' '.join([word for word in tt]) for tt in tgt]
        tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
            [' '.join(self.tokenizer.tokenize(tt)) for tt
             in tgt]) + ' [unused2]'

        raw_tgt = data['raw_tgt'][0]
        raw_tgt_subtokens_str = '[unused1] ' + ' [unused3] '.join(
            [' '.join(self.tokenizer.tokenize(tt)) for tt
             in raw_tgt]) + ' [unused2]'
        if self.args.split_qm:
            copy = data['copy']
            tgt_subtokens_str_out = '[unused1] '
            for t,c in zip(tgt,copy[0]):
                if c!=-1 and (self.args.dataset=="geo" or len(sents[c].split(' '))>1):
                    tgt_subtokens_str_out += sents[c].replace(' ','$$')+' '+' '.join(
                        self.tokenizer.tokenize(t)[1:]) + ' [unused3] '
                else:
                    tgt_subtokens_str_out += ' '.join(self.tokenizer.tokenize(t))+ ' [unused3] '
            tgt_subtokens_str_out = tgt_subtokens_str_out[:-11]+ ' [unused2]'
        else:

            tgt_subtokens_str_out = tgt_subtokens_str
        # src_tokens = src_tokens[:self.args.maxdoclen]
        # segments = segments[:self.args.maxdoclen]
        tgt_subtoken = tgt_subtokens_str.split(' ')[:self.args.max_tgt_len][:-1]+[self.tgt_eos]
        tgt_subtokens_out = tgt_subtokens_str_out.split(' ')[:self.args.max_tgt_len][:-1]+[self.tgt_eos]
        tgt_subtokens_out = [word.replace('$$',' ')  for word in tgt_subtokens_out]
        # tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        cls_ids = [i for i, t in enumerate(src_tokens) if t == self.cls_token]
        return tgt_subtoken,sents_pre,tgt_subtokens_out,raw_tgt_subtokens_str.split(),src_tokens,segments,cls_ids


