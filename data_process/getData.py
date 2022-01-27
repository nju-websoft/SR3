from dbutil import *
import evaluate
from rouge import Rouge
import numpy as np
import re
import json
rouge = Rouge()
def goldBestInfoWithRouge(infos,tgt,preinfo,prescore,goldinfoidxs):

    maxscore=prescore
    maxanswer=preinfo
    best_info_index=-1
    for idx,info in enumerate(infos):
        if idx in goldinfoidxs or len(info)==0:
            continue
        answer=preinfo+' '+info
        # print('a:',answer)
        # print('t:', tgt)
        scores = rouge.get_scores(answer, tgt)[0]
        scores = (scores['rouge-2']['f']+scores['rouge-1']['f'])/2.0
        if scores>maxscore:
            maxscore=scores
            best_info_index=idx
            maxanswer=answer
    return best_info_index,maxscore,maxanswer

def goldInfoWithRouge(infos,tgt):
    maxscore=0.09#0.16
    goldinfos=[]
    goldinfoidxs=[]
    answer=''
    maxlen=len(infos)
    for _ in range(maxlen):
        best_info_index, score,answer=goldBestInfoWithRouge(infos,tgt,answer,maxscore,goldinfoidxs)
        if best_info_index>=0 and score-maxscore>=0.005:
            # print(score)
            # print(infos[best_info_index])
            maxscore=score
            goldinfos.append(infos[best_info_index])
            goldinfoidxs.append(best_info_index)
        else:
            break
    return goldinfos,goldinfoidxs

def splitSentence(text,image=False):
    words = [word for word in text]
    sent = []
    sents = []
    if image:
        puncSplit = ['。','？', '!', '！',';','；']
    else:
        puncSplit = [',', ';',  '，', '；', '。', '；', '？', '：', '!', '！', '|',':']

    # puncSplit = ['。', '？', '！']
    for word in words:
        if word != '':
            sent.append(word)
        if word in puncSplit:
            sents.append(''.join(sent))
            sent = []
    if len(sent)>1:
        sents.append(''.join(sent))
    return sents

def add_geonames_bdbk(data):
    geonames=data['geonames']
    object2Geonames=data['object2Geonames']
    urlSummarys=data['urlSummarys']
    geonames_txt=[]
    entitys=[]
    final_climates={}
    for geoname in geonames:
        # 取气候类型，气候带最多的作为该地点的气候类型和气候带
        if len(geoname)>0:
            climates = {}
            hots = {}
            for single_geo in geoname:
                name = single_geo[1]
                climate=single_geo[-2]
                hot=single_geo[-1]
                if climate not in climates:
                    climates[climate]=0
                else:
                    climates[climate]+=1
                if hot not in hots:
                    hots[hot] = 0
                else:
                    hots[hot] += 1
            maxnum=-1
            final_climate=None
            for climate,num in climates.items():
                if num>maxnum:
                    maxnum=num
                    final_climate=climate
            entitys.append(name)
            if final_climate not in final_climates:
                final_climates[final_climate]=[name]
            else:
                if name not in final_climates[final_climate]:
                    final_climates[final_climate].append(name)
            maxnum = -1
            final_hot = None
            for hot, num in hots.items():
                if num > maxnum:
                    maxnum = num
                    final_hot = hot
            geonames_txt.append([(name+'为'+final_climate+'。',name),(name + '位于' + final_hot + '。', name)])
            # geonames_txt.append((name + '位于' + final_hot + '。', name))
    for name,item in object2Geonames.items():
        if name not in entitys:
            entitys.append(name)
            if item[-2] not in final_climates.keys():
                final_climates[item[-2]] = [name]
            else:
                if name not in final_climates[item[-2]]:
                    final_climates[item[-2]].append(name)
            geonames_txt.append([(name + '为' + item[-2] + '。', name),(name + '位于' + item[-1]+'。',name)])
            # geonames_txt.append((name + '为' + item[-2]+'。',name))
    summarys=[]

    senario_entities=[]
    for item in data['src']+data['material']:
        for sent,entity in item:
            for e in entity:
                if e not in senario_entities:
                    senario_entities.append(e)
    for item in data['image']+data['image_text']:
        for sent,entity in item[0]:
            for e in entity:
                if e not in senario_entities:
                    senario_entities.append(e)

    for urlSummary in urlSummarys:
        for single_urlSummary in urlSummary:
            url,name,summary = single_urlSummary
            if name in entitys or name in final_climates.keys():
                if name in final_climates.keys():
                    name=[name]+final_climates[name]
                else:
                    name=[name]
                summary = re.sub('{{([^}]+)::([^}]+)}}',r'\g<1>',summary)
                summarys.append((summary,name))
    newsummarys=[]
    for summary,name in summarys:
        newsummary=[[],name]
        for sent in splitSentence(summary):
            if len(sent)>1:
                sent=''.join(sent)
                sent_entity=[]
                for e in senario_entities:
                    if e in sent:
                        sent_entity.append(e)
                # if sent_entity!=[]:
                #     print(sent,sent_entity)
                newsummary[0].append([sent,sent_entity])
        newsummarys.append(newsummary)
    data['geonames']=geonames_txt
    data['urlSummarys']=newsummarys
    data['entities']=senario_entities
    data['climates']=final_climates

def recognize_entity(texts,ltp):
    entitys=[]
    new_texts=[]
    for text in texts:
        if isinstance(text,tuple):
            title = text[1]
            if title==None:
                title=""
            text=text[0]
        else: title=None
        if len(text)==0:
            continue
        # text = [''.join(temp) for temp in text if len(temp)>0]
        print(text)
        segs, hidden = ltp.seg(text)
        ners = ltp.ner(hidden)
        new_text=[]
        for ner,seg,t in zip(ners,segs,text):
            text_entitys=[]
            for tag, start, end in ner:
                tag = "".join(seg[start:end + 1])
                if tag not in text_entitys:
                    text_entitys.append(tag)
                # print(tag, ":", "".join(seg[0][start:end + 1]))
                if tag not in entitys:
                    entitys.append(tag)
            new_text.append((t,text_entitys))
        if title!=None:
            new_texts.append((new_text,title))
        else:
            new_texts.append(new_text)
    return entitys,new_texts

def findEntityWithJingwei(jingwei,geonames):
    results={}
    for name in jingwei:
        latitude,longitude = jingwei[name]["latitude"],jingwei[name]["longitude"]
        min_dis=1000000
        min_item=None
        for key,items in geonames.items():
            for item in items:
                if item[2]!=None and item[3]!=None:
                    la = float(item[2])
                    long = float(item[3])
                    dis=abs(la-latitude)+abs(long-longitude)
                    if dis<min_dis:
                        min_dis=dis
                        min_item=item
        results[name]=min_item
    return results

def addRetrieveData(readfile,writefile):
    from ltp import LTP
    ltp = LTP()
    geonames = getAllGeoNames()
    # geonames.sort(key=lambda k: (len(k[1])),reverse=True)
    with open(readfile, 'r', encoding='utf_8') as f:
        datas = json.load(f)
        for data in datas:
            qid = data["id"]
            print(qid)
            print(data)
            image_text=data['image_text']
            if image_text==[[]]:
                image_text=[]
            image = data['image']
            object2Geonames = findEntityWithJingwei(data['object2Geonames'],geonames)
            src = [[sent for sent in splitSentence(data['src']) if len(sent)>1]]
            entitys = []
            entity, src = recognize_entity(src, ltp)
            entitys.extend(entity)
            data['src'] = src
            tgt = [sent for sent in splitSentence(data['tgt']) if len(sent)>1]
            data['tgt'] = tgt
            material = [[sent for sent in splitSentence(text) if len(sent) > 1] for text in data['material']]
            entity, material = recognize_entity(material, ltp)
            entitys.extend(entity)
            data['material'] = material
            image = [([sent for sent in splitSentence(text[1], True) if len(sent) > 1],text[0]) for text in image]
            entity, image = recognize_entity(image, ltp)
            entitys.extend(entity)
            for single_image in image:
                for idx, (sent, entity_list) in enumerate(single_image[0]):
                    for key in object2Geonames.keys():
                        if key in sent:
                            if key not in entity_list:
                                entity_list.append(key)
                    single_image[0][idx] = (sent.replace('图示区域', ''), entity_list)

            data['image'] = image
            image_text = [([sent for sent in splitSentence(text[1]) if len(sent) > 1],text[0]) for text in
                          image_text]
            entity, image_text = recognize_entity(image_text, ltp)
            entitys.extend(entity)
            data['image_text'] = image_text
            current_geonames = []
            current_names = []
            urlSummarys = []
            for entity in entitys:
                # print(tag, ":", "".join(seg[0][start:end + 1]))
                if entity not in current_names:
                    current_names.append(entity)
            for name in current_names:
                if name in geonames:
                    urlSummary = getUrlSummaryByname(name)
                    if urlSummary:
                        urlSummarys.append(urlSummary)
                    current_geonames.append(geonames[name])
            climates = []
            for items in current_geonames:
                for item in items:
                    climate = item[-2]
                    if climate not in climates:
                        climates.append(climate)
            for key, item in object2Geonames.items():
                climate = item[-2]
                if climate not in climates:
                    climates.append(climate)
            for climate in climates:
                urlSummary = getUrlSummaryByname(climate)
                if urlSummary:
                    urlSummarys.append(urlSummary)

            # data["entities"]=entitys
            data["geonames"] = current_geonames
            data["object2Geonames"] = object2Geonames
            data["urlSummarys"] = urlSummarys
            add_geonames_bdbk(data)
            print()
    with open(writefile, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False)

def tagGoldInfo(readfile,writefile=None):
    with open(readfile,'r',encoding='utf-8') as f:
        datas = json.load(f)
    for data in datas:
        idx=0
        idx2tag={}
        tags={}
        tagsb=[]
        all_info_sents=[]
        tagsm = []
        for idx1, sents in enumerate(data['material']):
            temp = []
            for idx2, sent in enumerate(sents):
                all_info_sents.append(sent[0])
                temp.append(0)
                idx2tag[idx] = ('m', idx1, idx2)
                idx += 1
            tagsm.append(temp)
        tags['m']=tagsm
        if 'image' in data:
            tagsb = []
            for idx1, sents in enumerate(data['image']):
                temp = []
                for idx2, sent in enumerate(sents[0]):
                    all_info_sents.append(sent[0])
                    temp.append(0)
                    idx2tag[idx] = ('i', idx1, idx2)
                    idx += 1
                tagsb.append(temp)
            tags['i'] = tagsb
        if 'image_text' in data:
            tagsb = []
            for idx1, sents in enumerate(data['image_text']):
                temp = []
                for idx2, sent in enumerate(sents[0]):
                    all_info_sents.append(sent[0])
                    temp.append(0)
                    idx2tag[idx] = ('it', idx1, idx2)
                    idx += 1
                tagsb.append(temp)
        tags['it'] = tagsb
        tagsb = []
        for idx1, sents in enumerate(data['geonames']):
            temp = []
            for idx2, sent in enumerate(sents):
                all_info_sents.append(sent[0])
                temp.append(0)
                idx2tag[idx] = ('g', idx1, idx2)
                idx += 1
            tagsb.append(temp)
        tags['g'] = tagsb

        if 'new_geonames' in data:
            tagsb = []
            for idx1, sents in enumerate(data['new_geonames']):
                temp = []
                for idx2, sent in enumerate(sents):
                    all_info_sents.append(sent[0])
                    temp.append(0)
                    idx2tag[idx] = ('gn', idx1, idx2)
                    idx += 1
                tagsb.append(temp)
            tags['gn'] = tagsb

        tagsb = []
        for idx1, sents in enumerate(data['urlSummarys']):
            temp = []
            for idx2, sent in enumerate(sents[0]):
                all_info_sents.append(sent[0])
                temp.append(0)
                idx2tag[idx] = ('s', idx1, idx2)
                idx += 1
            tagsb.append(temp)
        tags['s'] = tagsb
        for idx in range(len(all_info_sents)):
            all_info_sents[idx]=evaluate.processText(all_info_sents[idx])
        tgt=evaluate.processText(''.join(data['tgt']))
        goldinfos, goldinfoidxs=goldInfoWithRouge(all_info_sents,tgt)
        for goldinfoidx in goldinfoidxs:
            type,idx1,idx2=idx2tag[goldinfoidx]
            tags[type][idx1][idx2]=1
        data['material_tag'] = tags['m']
        if 'i' in tags:
            data['image_tag'] = tags['i']
        if 'it' in tags:
            data['image_text_tag'] = tags['it']
        data['geonames_tag'] = tags['g']
        data['summary_tag'] = tags['s']

        if 'new_geonames' in data:
            data['new_geonames_tag'] = tags['gn']

        data['goldinfo']=goldinfos
        print('goldinfo:',goldinfos)
        print('tgt:',tgt)
    with open(writefile,'w',encoding='utf-8') as f:
        json.dump(datas,f,ensure_ascii=False)

def generateClassifyERNIEBysent(readfile,writefile):
    with open(readfile, 'r', encoding='utf_8') as f:
        with open(writefile, 'w', encoding='utf_8',newline='') as f_write:
            f_write.write('id	text_a	text_b	label\n')
            datas = json.load(f)
            for data in datas:
                question=''
                src = data['src']
                for doc in src:
                    for sent in doc:
                        question+=sent[0]
                material = data['material']
                image_text=data['image_text']
                image = data['image']
                # texts = background + material + image_text + image
                texts = material
                geonames = data['geonames']
                summarys = data['urlSummarys']
                material_tag = data['material_tag']
                image_tag = data['image_tag']
                image_text_tag = data['image_text_tag']
                geonames_tag = data['geonames_tag']
                summary_tag = data['summary_tag']
                texts = texts+geonames
                if 'new_geonames' in data:
                    geonames_ckgg = data['new_geonames']
                    texts=texts+geonames_ckgg
                    geonames_tag+=data['new_geonames_tag']
                split_len2 = len(texts)
                texts = texts+image_text+image
                split_len3 = len(texts)
                texts =texts+summarys
                tags = material_tag+geonames_tag+image_text_tag+image_tag+summary_tag
                assert len(texts)==len(tags)
                for idx,(doc,tag) in enumerate(zip(texts,tags)):
                    if idx>=split_len3:
                        assert len(doc[0]) == len(tag)
                        for sent, t in zip(doc[0], tag):
                            sent='#'.join(doc[1])+'$'+sent[0]
                            f_write.write(str(data['id'])+'\t'+question+'\t'+sent+'\t'+str(t)+'\n')
                            # writer.writerow([data['id'], question, sent, t])
                    elif idx>=split_len2:
                        assert len(doc[0]) == len(tag)
                        for sent, t in zip(doc[0], tag):
                            sent=doc[1]+'$'+sent[0]
                            f_write.write(str(data['id'])+'\t'+question+'\t'+sent+'\t'+str(t)+'\n')
                            # writer.writerow([data['id'], question, sent, t])
                    else:
                        assert len(doc) == len(tag)
                        for sent,t in  zip(doc,tag):
                            sent=sent[0]
                            f_write.write(str(data['id']) + '\t' + question + '\t' + sent + '\t' + str(t) + '\n')
                            # writer.writerow([data['id'], question, sent, t])

def reloadERNIEscore(raw_file,ernie_file,pred_file,write_file):
    pred_results=[]
    with open(ernie_file,'r',encoding='utf-8') as f1:
        f1.readline()
        with open(pred_file, 'r', encoding='utf-8') as f2:
            for data,pred in zip(f1,f2):
                id,text_a,text_b,label=data.split('\t')
                score = pred.strip().split('	')[-1][1:-1].strip(' ').split(' ')[-1]
                pred_results.append((text_b,float(score)))
    pred_idx=0
    with open(raw_file, 'r', encoding='utf_8') as f:
        with open(write_file, 'w', encoding='utf_8',newline='') as f_write:
            datas = json.load(f)
            for data in datas:
                question=''
                src = data['src']
                for doc in src:
                    for sent in doc:
                        question+=sent[0]
                material = data['material']
                image_text=data['image_text']
                geonames = data['geonames']
                if 'new_geonames' in data:
                    geonames_new = data['new_geonames']

                image = data['image']
                summarys = data['urlSummarys']
                material_scores=[]
                for idx,doc in enumerate(material):
                    for sent in doc:
                        sent=sent[0]
                        text_b,score = pred_results[pred_idx]
                        pred_idx += 1
                        if sent!=text_b:
                            print(sent,text_b)
                        material_scores.append(score)
                geonames_scores = []
                for idx, doc in enumerate(geonames):
                    for sent in doc:
                        sent = sent[0]
                        text_b, score = pred_results[pred_idx]
                        pred_idx += 1
                        if sent != text_b:
                            print(sent, text_b)
                        geonames_scores.append(score)
                if 'new_geonames' in data:
                    new_geonames_scores = []
                    for idx, doc in enumerate(geonames_new):
                        for sent in doc:
                            sent = sent[0]
                            text_b, score = pred_results[pred_idx]
                            pred_idx += 1
                            if sent != text_b:
                                print(sent, text_b)
                            new_geonames_scores.append(score)

                image_text_scores=[]
                for idx,doc in enumerate(image_text):
                    for sent in doc[0]:
                        sent=sent[0]
                        text_b,score = pred_results[pred_idx]
                        pred_idx += 1
                        text_b = text_b.split('$')[-1]
                        if sent!=text_b:
                            print(sent,text_b)
                        image_text_scores.append(score)
                image_scores=[]
                for idx, doc in enumerate(image):
                    for sent in doc[0]:
                        sent = sent[0]
                        text_b, score = pred_results[pred_idx]
                        pred_idx += 1
                        text_b = text_b.split('$')[-1]
                        if sent != text_b:
                            print(sent, text_b)
                        image_scores.append(score)
                summary_scores=[]
                for idx, doc in enumerate(summarys):
                    for sent in doc[0]:
                        sent = sent[0]
                        text_b, score = pred_results[pred_idx]
                        pred_idx += 1
                        text_b=text_b.split('$')[-1]
                        if sent != text_b:
                            print(sent, text_b)
                        summary_scores.append(score)
                data['material_scores'] = material_scores
                data['image_text_scores'] = image_text_scores
                data['image_scores'] = image_scores
                data['geonames_scores'] = geonames_scores
                if 'new_geonames' in data:
                    data['new_geonames_scores'] = new_geonames_scores
                data['summary_scores'] = summary_scores
            json.dump(datas,f_write,ensure_ascii=False)


def generateGraphfile(raw_file,writefile):
    with open(raw_file,'r',encoding='utf-8') as f:
        datas=json.load(f)
        newdatas=[]
        for data in datas:
            question = ''
            src = data['src']
            q_entities=[]
            for doc in src:
                for sent in doc:
                    question += sent[0]
                    q_entities.extend(sent[1])
            material_scores=data['material_scores']
            geonames_scores=data['geonames_scores'] if 'new_geonames' not in data else data['geonames_scores']+data['new_geonames_scores']
            image_text_scores=data['image_text_scores']
            image_scores=data['image_scores']
            summary_scores=data['summary_scores']
            scores=material_scores+geonames_scores+image_text_scores+image_scores+summary_scores
            scores_sort=list(np.argsort(-np.array(scores)))
            scores_idx=[0]*len(scores_sort)
            for idx,sort in enumerate(scores_sort):
                scores_idx[sort]=idx+1
            # scores_idx+=1
            geonames = data['geonames'] if 'new_geonames' not in data else data['geonames']+data['new_geonames']
            material = data['material']
            image_text = data['image_text']
            image = data['image']
            summarys = data['urlSummarys']
            geonames_tag = data['geonames_tag'] if 'new_geonames' not in data else data['geonames_tag']+data['new_geonames_tag']
            material_tag = data['material_tag']
            image_tag = data['image_tag']
            image_text_tag = data['image_text_tag']
            summary_tag = data['summary_tag']
            tags=[]
            for t_list in material_tag+geonames_tag + image_text_tag + image_tag + summary_tag:
                tags.extend(t_list)
            assert len(tags)==len(scores)
            node=[0]*(len(scores)+1)
            node[0]=question
            node_type=['']*(len(scores)+1)
            node_type[0]='question'
            relations = [[None for i in range(len(scores)+1)] for j in range(len(scores)+1)]
            same_doc_sents=[]
            entity_map={}
            idx=0
            for doc in material:
                sent_idx = []
                for sent in doc:
                    text = sent[0]
                    sent_idx.append(scores_idx[idx])
                    node[scores_idx[idx]] = text
                    node_type[scores_idx[idx]] = 'material'
                    relations[0][scores_idx[idx]] = 'q_senario'
                    relations[scores_idx[idx]][0] = 'senario_q'
                    entities = sent[1]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[scores_idx[idx]]
                        else:
                            entity_map[e].append(scores_idx[idx])
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in geonames:
                sent_idx=[]
                for sent in doc:
                    text = sent[0]
                    sent_idx.append(scores_idx[idx])
                    node[scores_idx[idx]]=text
                    node_type[scores_idx[idx]]='geonames'
                    entities = [sent[1]]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[scores_idx[idx]]
                        else:
                            entity_map[e].append(scores_idx[idx])
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in image_text:
                sent_idx = []
                for sent in doc[0]:
                    text = sent[0]
                    sent_idx.append(scores_idx[idx])
                    node[scores_idx[idx]] = text
                    node_type[scores_idx[idx]] = 'image_text'
                    relations[0][scores_idx[idx]] = 'q_senario'
                    relations[scores_idx[idx]][0] = 'senario_q'
                    entities = sent[1]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[scores_idx[idx]]
                        else:
                            entity_map[e].append(scores_idx[idx])
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in image:
                sent_idx = []
                for sent in doc[0]:
                    text = sent[0]
                    sent_idx.append(scores_idx[idx])
                    node[scores_idx[idx]] = text
                    node_type[scores_idx[idx]] = 'image'
                    relations[0][scores_idx[idx]] = 'q_senario'
                    relations[scores_idx[idx]][0] = 'senario_q'
                    entities = sent[1]
                    for e in entities:
                        e=e.replace(" ",'')
                        if e not in entity_map:
                            entity_map[e]=[scores_idx[idx]]
                        else:
                            entity_map[e].append(scores_idx[idx])
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in summarys:
                doc_entities=doc[1]
                sent_idx = []
                for sent in doc[0]:
                    text = sent[0]
                    sent_idx.append(scores_idx[idx])
                    node[scores_idx[idx]] = text
                    node_type[scores_idx[idx]] = 'summary'
                    entities = sent[1]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[scores_idx[idx]]
                        else:
                            entity_map[e].append(scores_idx[idx])
                    idx+=1
                for e in doc_entities:
                    e = e.replace(" ", '')
                    if e not in entity_map:
                        entity_map[e]=sent_idx
                    else:
                        entity_map[e].extend(sent_idx)
                same_doc_sents.append(sent_idx)
            for e in q_entities:
                e = e.replace(" ", '')
                if e not in entity_map:
                    entity_map[e] = [0]
                else:
                    entity_map[e].append(0)
            for entitiy,item in entity_map.items():
                for idxs1 in item:
                    for idxs2 in item:
                        if idxs2!=idxs1:
                            relations[idxs1][idxs2] = 'sameEntity'
            for sent_idx in same_doc_sents:
                for idxs1 in sent_idx:
                    for idxs2 in sent_idx:
                        if idxs2 != idxs1:
                            relations[idxs1][idxs2] = 'samedoc'
            node_labels=[-1]*len(node)
            for idx,label in enumerate(tags):
                node_labels[scores_idx[idx]]=label

            newdata={}
            newdata['id']=data['id']
            newdata['nodes']=node
            newdata['node_types']=node_type
            newdata['relations']=relations
            newdata['node_labels']=node_labels
            newdata['nodes_raworder'] = str([0] + [x + 1 for x in scores_sort])
            newdata['tgt']=[data['tgt']]
            newdatas.append(newdata)
    with open(writefile,'w',encoding='utf-8') as f:
        json.dump(newdatas,f,ensure_ascii=False)


def generateGraphfileWithoutRank(raw_file,writefile):
    with open(raw_file,'r',encoding='utf-8') as f:
        datas=json.load(f)
        newdatas=[]
        for data in datas:
            question = ''
            src = data['src']
            q_entities=[]
            for doc in src:
                for sent in doc:
                    question += sent[0]
                    q_entities.extend(sent[1])
            geonames = data['geonames']
            material = data['material']
            image_text = data['image_text']
            image = data['image']
            summarys = data['urlSummarys']
            geonames_tag = data['geonames_tag']
            material_tag = data['material_tag']
            image_tag = data['image_tag']
            image_text_tag = data['image_text_tag']
            summary_tag = data['summary_tag']
            tags=[]
            for t_list in image_text_tag + image_tag + material_tag+geonames_tag +summary_tag:
                tags.extend(t_list)
            node=[0]*(len(tags)+1)
            node[0]=question
            node_type=['']*(len(tags)+1)
            node_type[0]='question'
            relations = [[None for i in range(len(tags)+1)] for j in range(len(tags)+1)]
            same_doc_sents=[]
            entity_map={}
            idx=1
            for doc in image_text:
                sent_idx = []
                for sent in doc[0]:
                    text = sent[0]
                    sent_idx.append(idx)
                    node[idx] = text
                    node_type[idx] = 'image_text'
                    relations[0][idx] = 'q_senario'
                    relations[idx][0] = 'senario_q'
                    entities = sent[1]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[idx]
                        else:
                            entity_map[e].append(idx)
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in image:
                sent_idx = []
                for sent in doc[0]:
                    text = sent[0]
                    sent_idx.append(idx)
                    node[idx] = text
                    node_type[idx] = 'image'
                    relations[0][idx] = 'q_senario'
                    relations[idx][0] = 'senario_q'
                    entities = sent[1]
                    for e in entities:
                        e=e.replace(" ",'')
                        if e not in entity_map:
                            entity_map[e]=[idx]
                        else:
                            entity_map[e].append(idx)
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in material:
                sent_idx = []
                for sent in doc:
                    text = sent[0]
                    sent_idx.append(idx)
                    node[idx] = text
                    node_type[idx] = 'material'
                    relations[0][idx] = 'q_senario'
                    relations[idx][0] = 'senario_q'
                    entities = sent[1]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[idx]
                        else:
                            entity_map[e].append(idx)
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in geonames:
                sent_idx=[]
                for sent in doc:
                    text = sent[0]
                    sent_idx.append(idx)
                    node[idx]=text
                    node_type[idx]='geonames'
                    entities = [sent[1]]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[idx]
                        else:
                            entity_map[e].append(idx)
                    idx+=1
                same_doc_sents.append(sent_idx)
            for doc in summarys:
                doc_entities=doc[1]
                sent_idx = []
                for sent in doc[0]:
                    text = sent[0]
                    sent_idx.append(idx)
                    node[idx] = text
                    node_type[idx] = 'summary'
                    entities = sent[1]
                    for e in entities:
                        e = e.replace(" ", '')
                        if e not in entity_map:
                            entity_map[e]=[idx]
                        else:
                            entity_map[e].append(idx)
                    idx+=1
                for e in doc_entities:
                    e = e.replace(" ", '')
                    if e not in entity_map:
                        entity_map[e]=sent_idx
                    else:
                        entity_map[e].extend(sent_idx)
                same_doc_sents.append(sent_idx)
            for e in q_entities:
                e = e.replace(" ", '')
                if e not in entity_map:
                    entity_map[e] = [0]
                else:
                    entity_map[e].append(0)
            for entitiy,item in entity_map.items():
                for idxs1 in item:
                    for idxs2 in item:
                        if idxs2!=idxs1:
                            relations[idxs1][idxs2] = 'sameEntity'
            for sent_idx in same_doc_sents:
                for idxs1 in sent_idx:
                    for idxs2 in sent_idx:
                        if idxs2 != idxs1:
                            relations[idxs1][idxs2] = 'samedoc'
            node_labels=[-1]+tags
            newdata={}
            newdata['id']=data['id']
            newdata['nodes']=node
            newdata['node_types']=node_type
            newdata['relations']=relations
            newdata['node_labels']=node_labels
            newdata['nodes_raworder'] = str([i for i in range(len(node))])
            newdata['tgt']=[data['tgt']]
            newdatas.append(newdata)
    with open(writefile,'w',encoding='utf-8') as f:
        json.dump(newdatas,f,ensure_ascii=False)

def calmetricERNIE(truefile,pred_file,writefile):
    with open(truefile,'r',encoding='utf-8') as f_true:
        with open(pred_file,'r',encoding='utf-8') as f_pred:
            with open(writefile, 'w', encoding='utf-8') as f_write:
                f_true.readline()
                labels=[]
                scores=[]
                sents=[]
                oldqid=None
                metrics=[]
                for idx,(true_line,pred_line) in enumerate(zip(f_true,f_pred)):
                    pred_line=pred_line.strip().split('\t')
                    true_line = true_line.strip().split('\t')
                    qid=true_line[0]
                    if oldqid==None:
                        oldqid=qid
                    if qid!=oldqid and oldqid!=None:
                        pred_idx = []
                        gold_idx = []
                        result=''
                        selected_ids = np.argsort(scores, 0)
                        for i, idx in enumerate(selected_ids):
                            result+=sents[idx].split('$')[-1]
                            # if len(result)<300:
                            pred_idx.append(idx)
                            if labels[idx] == 1:
                                gold_idx.append(idx)
                        metric=evaluate.cal_metrics(gold_idx,pred_idx,['MAP', 'NDCG', 'HIT'],maxnum=5)
                        # print(result)
                        f_write.write(result+'\n')
                        metrics.append(metric)
                        labels = []
                        scores = []
                        oldqid = qid
                        sents=[]
                    true_label=int(true_line[-1])
                    sents.append(true_line[-2])
                    # print(pred_line[-1].replace("[","").replace("]","").split(" ")[-1])
                    pred_score = float(pred_line[-1].replace("[","").replace("]","").strip(" ").split(" ")[-1])
                    # pred_score = float(pred_line[-1].replace("[", "").replace("]", "").strip(" ").split("\t")[-1])
                    labels.append(true_label)
                    scores.append(-pred_score)
                pred_idx=[]
                gold_idx=[]
                result = ''
                selected_ids = np.argsort(scores, 0)
                for i, idx in enumerate(selected_ids):
                    # if len(result) < 300:
                    pred_idx.append(idx)
                    result += sents[idx].split('$')[-1]
                    if labels[idx]==1:
                        gold_idx.append(idx)
                metric = evaluate.cal_metrics(gold_idx, pred_idx, ['MAP', 'NDCG', 'HIT'], maxnum=5)
                # print(result)
                f_write.write(result + '\n')
                metrics.append(metric)
                scoreAvg = np.mean(metrics, axis=0)
                for metric, score in zip(['MAP', 'NDCG', 'HIT'], scoreAvg):
                    print(metric,':',score)

# if __name__ == '__main__':
    # addRetrieveData('../data/train_raw.json', '../data/train_retrieve_temp.json')
    # addRetrieveData('../data/val_raw.json', '../data/val_retrieve_temp.json')
    # addRetrieveData('../data/test_raw.json', '../data/test_retrieve_temp.json')

    # tagGoldInfo('../data/test_retrieve.json','../data/test.json')
    # tagGoldInfo('../data/val_retrieve.json','../data/val.json')
    # tagGoldInfo('../data/train_retrieve.json','../data/train.json')

    # generateClassifyERNIEBysent('../data/train.json','../data/train.tsv')
    # generateClassifyERNIEBysent('../data/val.json','../data/val.tsv')
    # generateClassifyERNIEBysent('../data/test.json','../data/test.tsv')

    # reloadERNIEscore('../data/val.json',
    #                  '../data/val.tsv',
    #                  '../data/val_result.0.0',
    #                  '../data/val_score.json')
    # reloadERNIEscore('../data/test.json',
    #                  '../data/test.tsv',
    #                  '../data/test_result.0.0',
    #                  '../data/test_score.json')
    # reloadERNIEscore('../data/train.json',
    #                  '../data/train.tsv',
    #                  '../data/train_result.0.0',
    #                  '../data/train_score.json')

    # generateGraphfile('../data/val_score.json',
    #                   '../data/val_graph.json')
    # generateGraphfile('../data/test_score.json',
    #                   '../data/test_graph.json')
    # generateGraphfile('../data/train_score.json',
    #                   '../data/train_graph.json')

    # calmetricERNIE('../data/test.tsv',
    #                '../data/test_result.0.0',
    #                '../data/test.txt')
