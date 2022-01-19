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

# import xlrd, xlwt
# def writeImageData(filename,qas):
#     with open(filename,'w',encoding='utf-8') as f_write:
#         write_data=[]
#         for id, background_id,question, answer, material,image_text, image, object2Geonames in qas:
#             question = removeunuseful(question)
#             answer = removeunuseful(answer)
#             material = [removeunuseful(m) for m  in material]
#             image = [(m[0],removeunuseful(m[1])) for m in image]
#             image_text = [(m[0],removeunuseful(m[1])) for m in image_text]
#             if image_text==[]:
#                 image_text=[[]]
#             write_data.append({'id':id,'background_id':background_id,'src':question,'tgt':answer,'material':material,'image':image,'image_text':image_text,'object2Geonames':object2Geonames})
#         f_write.write(json.dumps(write_data,ensure_ascii=False))

# def writeImageData_splitReason(filename,qas):
#     with open(filename,'w',encoding='utf-8') as f_write:
#         write_data=[]
#         for id, background_id,question, answer, material,image_text, image,image_reason, object2Geonames in qas:
#             question = removeunuseful(question)
#             answer = removeunuseful(answer)
#             material = [removeunuseful(m) for m  in material]
#             image = [(m[0],removeunuseful(m[1])) for m in image]
#             image_text = [(m[0],removeunuseful(m[1])) for m in image_text]
#             image_reason = [(m[0], removeunuseful(m[1])) for m in image_reason]
#             if image_text==[]:
#                 image_text=[[]]
#             write_data.append({'id':id,'background_id':background_id,'src':question,'tgt':answer,'material':material,'image':image,'image_text':image_text,'image_reason':image_reason,'object2Geonames':object2Geonames})
#         f_write.write(json.dumps(write_data,ensure_ascii=False))

# def writeImageDataALLNew(writefile):
#     qas = getImageAnnotatedQANew()
#     print(len(qas))
#     count=0
#     temp = image_id_cache
#     for key,value in temp.items():
#         if value[0] and value[1]==None:
#             count+=1
#     print(count)
#     writeImageData(writefile, qas)
#
# def writeImageDataALLNew_splitReason(writefile):
#     qas = getImageAnnotatedQANew_splitreason()
#     writeImageData_splitReason(writefile, qas)

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

# def getTrainTestRaw(directory_path,write_path):
#     raw_data={}
#     with open(directory_path+"image_data.json","r",encoding="utf-8") as f:
#         json_data=json.load(f)
#         for data in json_data:
#             raw_data[data['id']]=data
#     train_data=[]
#     test_data=[]
#     val_data=[]
#     with open(directory_path+"train.json","r",encoding="utf-8") as f:
#         json_data = json.load(f)
#         for data in json_data:
#             train_data.append(raw_data[data['id']])
#     with open(directory_path + "val.json", "r", encoding="utf-8") as f:
#         json_data = json.load(f)
#         for data in json_data:
#             val_data.append(raw_data[data['id']])
#     with open(directory_path + "test.json", "r", encoding="utf-8") as f:
#         json_data = json.load(f)
#         for data in json_data:
#             test_data.append(raw_data[data['id']])
#     with open(write_path + "train_raw.json", "w", encoding="utf-8") as f:
#         json.dump(train_data,f,ensure_ascii=False)
#     with open(write_path + "val_raw.json", "w", encoding="utf-8") as f:
#         json.dump(val_data, f, ensure_ascii=False)
#     with open(write_path + "test_raw.json", "w", encoding="utf-8") as f:
#         json.dump(test_data, f, ensure_ascii=False)
#

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
            object2Geonames = data['object2Geonames']
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

# def splitTrainTest(readfile,writepath):
#     with open(readfile, 'r', encoding='utf_8') as f:
#         datas = json.load(f)
#         background_ids=[]
#         for data in datas:
#             bid = data['background_id']
#             if bid not in background_ids:
#                 background_ids.append(bid)
#         random.shuffle(background_ids)
#         train_bid = background_ids[:int(len(background_ids)*0.6)]
#         val_bid = background_ids[int(len(background_ids)*0.6):int(len(background_ids)*0.8)]
#         test_bid = background_ids[int(len(background_ids)*0.8):]
#         train_datas=[]
#         val_datas=[]
#         test_datas = []
#         for data in datas:
#             bid = data['background_id']
#             if bid in train_bid:
#                 train_datas.append(data)
#             elif bid in val_bid:
#                 val_datas.append(data)
#             else:
#                 test_datas.append(data)
#     with open(writepath+"/train.json",'w',encoding='utf-8') as f:
#         json.dump(train_datas, f, ensure_ascii=False)
#     with open(writepath+"/val.json",'w',encoding='utf-8') as f:
#         json.dump(val_datas, f, ensure_ascii=False)
#     with open(writepath+"/test.json",'w',encoding='utf-8') as f:
#         json.dump(test_datas, f, ensure_ascii=False)
#
# def calmetricERNIE(truefile,pred_file,writefile):
#     with open(truefile,'r',encoding='utf-8') as f_true:
#         with open(pred_file,'r',encoding='utf-8') as f_pred:
#             with open(writefile, 'w', encoding='utf-8') as f_write:
#                 f_true.readline()
#                 labels=[]
#                 scores=[]
#                 sents=[]
#                 oldqid=None
#                 metrics=[]
#                 for idx,(true_line,pred_line) in enumerate(zip(f_true,f_pred)):
#                     pred_line=pred_line.strip().split('\t')
#                     true_line = true_line.strip().split('\t')
#                     qid=true_line[0]
#                     if oldqid==None:
#                         oldqid=qid
#                     if qid!=oldqid and oldqid!=None:
#                         pred_idx = []
#                         gold_idx = []
#                         result=''
#                         selected_ids = np.argsort(scores, 0)
#                         for i, idx in enumerate(selected_ids):
#                             result+=sents[idx].split('$')[-1]
#                             # if len(result)<300:
#                             pred_idx.append(idx)
#                             if labels[idx] == 1:
#                                 gold_idx.append(idx)
#                         metric=evaluate.cal_metrics(gold_idx,pred_idx,['MAP', 'NDCG', 'HIT'],maxnum=5)
#                         # print(result)
#                         f_write.write(result+'\n')
#                         metrics.append(metric)
#                         labels = []
#                         scores = []
#                         oldqid = qid
#                         sents=[]
#                     true_label=int(true_line[-1])
#                     sents.append(true_line[-2])
#                     # print(pred_line[-1].replace("[","").replace("]","").split(" ")[-1])
#                     pred_score = float(pred_line[-1].replace("[","").replace("]","").strip(" ").split(" ")[-1])
#                     # pred_score = float(pred_line[-1].replace("[", "").replace("]", "").strip(" ").split("\t")[-1])
#                     labels.append(true_label)
#                     scores.append(-pred_score)
#                 pred_idx=[]
#                 gold_idx=[]
#                 result = ''
#                 selected_ids = np.argsort(scores, 0)
#                 for i, idx in enumerate(selected_ids):
#                     # if len(result) < 300:
#                     pred_idx.append(idx)
#                     result += sents[idx].split('$')[-1]
#                     if labels[idx]==1:
#                         gold_idx.append(idx)
#                 metric = evaluate.cal_metrics(gold_idx, pred_idx, ['MAP', 'NDCG', 'HIT'], maxnum=5)
#                 # print(result)
#                 f_write.write(result + '\n')
#                 metrics.append(metric)
#                 scoreAvg = np.mean(metrics, axis=0)
#                 for metric, score in zip(['MAP', 'NDCG', 'HIT'], scoreAvg):
#                     print(metric,':',score)
#
# def getanswer(readfile,writefile):
#     with open(readfile,'r',encoding='utf-8') as f:
#         with open(writefile, 'w', encoding='utf-8') as f_w:
#             datas = json.load(f)
#             for data in datas:
#                 f_w.write(''.join(data['tgt'])+'\n')

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

# def cal_statistics(readfile):
#     statistics={
#         "question_len":[],
#         "answer_len":[],
#         "answer_sent_size":[],
#         "text_scenario_len":[],
#         "text_scenario_sent_size":[],
#         "image_scenario_len": [],
#         "image_scenario_sent_size": [],
#         "scenario_len": [],
#         "scenario_sent_size": [],
#     }
#     with open(readfile,'r',encoding='utf-8') as f:
#         datas = json.load(f)
#         for data in datas:
#             question = []
#             for doc in data['src']:
#                 for sent in doc:
#                     question += sent[0]
#             material = []
#             for idx1, sents in enumerate(data['material']):
#                 for idx2, sent in enumerate(sents):
#                     material.append(sent[0])
#             image = []
#             if 'image' in data:
#                 for idx1, sents in enumerate(data['image']):
#                     for idx2, sent in enumerate(sents[0]):
#                         image.append(sent[0])
#             if 'image_text' in data:
#                 for idx1, sents in enumerate(data['image_text']):
#                     for idx2, sent in enumerate(sents[0]):
#                         image.append(sent[0])
#             statistics['question_len'].append(len(''.join(question)))
#             statistics['answer_len'].append(len(evaluate.processText(''.join(data['tgt']), replace=True).split()))
#             statistics['answer_sent_size'].append(len(data['tgt']))
#             statistics['text_scenario_len'].append(len(''.join(material)))
#             statistics['text_scenario_sent_size'].append(len(material))
#             statistics['image_scenario_len'].append(len(''.join(image)))
#             statistics['image_scenario_sent_size'].append(len(image))
#             statistics['scenario_len'].append(len(''.join(material))+len(''.join(image)))
#             statistics['scenario_sent_size'].append(len(material)+len(image))
#
#     for key, value in statistics.items():
#         statisticsAvg = np.mean(value)
#         statisticsStd = np.std(value)
#         print(key, statisticsAvg,statisticsStd)
#         print(np.min(value),np.max(value))

# def splitsentence(readfile):
#     statistics = {
#         "text_scenario_sent_size": [],
#         "image_len": [],
#         "image_manual_sent_size": [],
#         "image_reason_sent_size": [],
#     }
#     with open(readfile,'r',encoding='utf-8') as f:
#         datas = json.load(f)
#         for data in datas:
#             data['material'] = [[sent for sent in splitSentence(text) if len(sent) > 1] for text in data['material']]
#             material = []
#             for idx1, sents in enumerate(data['material']):
#                 for idx2, sent in enumerate(sents):
#                     material.append(sent)
#             image = []
#             if 'image' in data:
#                 data['image'] = [([sent for sent in splitSentence(text[1], True) if len(sent) > 1], text[0]) for text in
#                          data['image']]
#                 for idx1, sents in enumerate( data['image']):
#                     for idx2, sent in enumerate(sents[0]):
#                         image.append(sent)
#             if 'image_text' in data and data['image_text']!=[[]]:
#                 image_text = [([sent for sent in splitSentence(text[1]) if len(sent) > 1], text[0]) for text in
#                               data['image_text']]
#                 for idx1, sents in enumerate(image_text):
#                     for idx2, sent in enumerate(sents[0]):
#                         image.append(sent)
#             image_reason=[]
#             if 'image_reason' in data:
#                 image_reason_text = [([sent for sent in splitSentence(text[1]) if len(sent) > 1], text[0]) for text in
#                               data['image_reason']]
#                 for idx1, sents in enumerate(image_reason_text):
#                     for idx2, sent in enumerate(sents[0]):
#                         image_reason.append(sent)
#             statistics['text_scenario_sent_size'].append(len(material))
#             statistics['image_len'].append(len(data['image']))
#             statistics['image_manual_sent_size'].append(len(image))
#             statistics['image_reason_sent_size'].append(len(image_reason))
#     for key, value in statistics.items():
#         statisticsAvg = np.mean(value)
#         statisticsStd = np.std(value)
#         print(key, statisticsAvg,statisticsStd)
#         print(np.min(value),np.max(value))
#
# def sampleForHuman(readfile,writepath):
#     import os
#     with open(readfile,'r',encoding='utf-8') as f:
#         datas=json.load(f)
#         data_dict={}
#         for data in datas:
#             id = data['id']
#             background_id=data['background_id']
#             question = ''
#             for doc in data['src']:
#                 for sent in doc:
#                     question += sent[0]
#             answer=''.join(data['tgt'])
#             if background_id in data_dict:
#                 data_dict[background_id].append((id,question,answer))
#             else:
#                 data_dict[background_id]=[(id, question, answer)]
#         data_list=list(data_dict)
#         idxs=list(range(len(data_list)))
#         q_nums=0
#         human_datas=[]
#         while q_nums<100:
#             idx=random.choice(idxs)
#             idxs.remove(idx)
#             human_datas.append(data_list[idx])
#             q_nums+=len(data_dict[data_list[idx]])
#         human_datas_final=[]
#         image_path=writepath+'/images/'
#         for human_data in human_datas:
#             temp_data={}
#             temp_data['background_id']=human_data
#             temp_data['qids']=[qid for qid,q,a in data_dict[human_data]]
#             bg_material, materials=get_material(temp_data['qids'])
#             image_ids=get_image_ids(temp_data['qids'])
#             temp_data['image_ids']=image_ids
#             temp_data['bg_materials']=bg_material
#             temp_data['questions']=[]
#             assert len(materials)==len(data_dict[human_data])
#             for m,(qid,q,a) in zip(materials,data_dict[human_data]):
#                 temp_data['questions'].append({
#                     'question':q,
#                     'answer':a,
#                     'material':m
#                 })
#             isExists = os.path.exists(image_path+str(temp_data['background_id']))
#             if not isExists:
#                 os.mkdir(image_path+str(temp_data['background_id']))
#             for idx,image_id in enumerate(image_ids):
#                 image=read_image(image_id)
#                 with open(image_path+str(temp_data['background_id'])+'/'+str(image_id)+'.jpg', 'wb') as f:
#                     f.write(image)
#             human_datas_final.append(temp_data)
#         with open(writepath+'/human_performance.json','w',encoding='utf-8') as f:
#             json.dump(human_datas_final,f,ensure_ascii=False)
#
# def sampleForHumanWithout(readfile,writepath,exsitedfile):
#     import os
#     exsited_bg=[]
#     with open(exsitedfile,'r',encoding='utf-8') as f:
#         datas=json.load(f)
#         for data in datas:
#             exsited_bg.append(data['background_id'])
#     with open(readfile,'r',encoding='utf-8') as f:
#         datas=json.load(f)
#         data_dict={}
#         for data in datas:
#             id = data['id']
#             background_id=data['background_id']
#             if background_id in exsited_bg:
#                 continue
#             question = ''
#             for doc in data['src']:
#                 for sent in doc:
#                     question += sent[0]
#             answer=''.join(data['tgt'])
#             if background_id in data_dict:
#                 data_dict[background_id].append((id,question,answer))
#             else:
#                 data_dict[background_id]=[(id, question, answer)]
#         data_list=list(data_dict)
#         idxs=list(range(len(data_list)))
#         q_nums=0
#         human_datas=[]
#         while q_nums<50:
#             idx=random.choice(idxs)
#             idxs.remove(idx)
#             human_datas.append(data_list[idx])
#             q_nums+=len(data_dict[data_list[idx]])
#         human_datas_final=[]
#         image_path=writepath+'/images/'
#         for human_data in human_datas:
#             temp_data={}
#             temp_data['background_id']=human_data
#             temp_data['qids']=[qid for qid,q,a in data_dict[human_data]]
#             bg_material, materials=get_material(temp_data['qids'])
#             image_ids=get_image_ids(temp_data['qids'])
#             temp_data['image_ids']=image_ids
#             temp_data['bg_materials']=bg_material
#             temp_data['questions']=[]
#             assert len(materials)==len(data_dict[human_data])
#             for m,(qid,q,a) in zip(materials,data_dict[human_data]):
#                 temp_data['questions'].append({
#                     'question':q,
#                     'answer':a,
#                     'material':m
#                 })
#             isExists = os.path.exists(image_path+str(temp_data['background_id']))
#             if not isExists:
#                 os.mkdir(image_path+str(temp_data['background_id']))
#             for idx,image_id in enumerate(image_ids):
#                 image=read_image(image_id)
#                 with open(image_path+str(temp_data['background_id'])+'/'+str(image_id)+'.jpg', 'wb') as f:
#                     f.write(image)
#             human_datas_final.append(temp_data)
#         with open(writepath+'/human_performance.json','w',encoding='utf-8') as f:
#             json.dump(human_datas_final,f,ensure_ascii=False)
#
# def get_SR3RES_humanperformance(SR3file,humanfile,testfile,writefile):
#     with open(SR3file,'r',encoding='utf-8') as f:
#         answer=f.readlines()
#     answermap={}
#     with open(testfile,'r',encoding='utf-8') as f:
#         datas=json.load(f)
#         for idx,data in enumerate(datas):
#             aid=data['id']
#             answermap[aid]=answer[idx]
#     SR3res=[]
#     exsisted_id=[]
#     with open(humanfile,'r',encoding='utf-8') as f:
#         datas=json.load(f)
#         for data in datas:
#             for qid in data['qids']:
#                 if qid in exsisted_id:
#                     print(qid)
#                 else:
#                     exsisted_id.append(qid)
#                 SR3res.append({"qid:":qid,"answer":answermap[qid].strip()})
#     print(len(exsisted_id))
#     with open(writefile,'w',encoding='utf-8') as f:
#         json.dump(SR3res,f,ensure_ascii=False)
#
# def change_qa_with_score(readfile,writefile):
#     with open(readfile,'r',encoding='utf-8') as f:
#         datas=json.load(f)
#     for data in datas:
#         qas=get_question_answer(data['qids'])
#         for item,qa in zip(data['questions'],qas):
#             print(item)
#             print(qa)
#             item['question']=qa[0]
#             item['answer']=qa[1]
#     with open(writefile,'w',encoding='utf-8') as f:
#         json.dump(datas,f,ensure_ascii=False)
#

# def combineBaselinesGSQAWithHuman(testfile,humanfile,base_dir,mymethod,writefile):
#
#
#     with open(testfile,'r',encoding='utf-8') as f:
#         test_datas = json.load(f)
#     gold_answers={}
#     for data in test_datas:
#         gold_answers[data['id']]="".join(data['tgt'])
#
#     humandata = xlrd.open_workbook(humanfile)
#     humandata = humandata.sheet_by_index(0)
#     nrows = humandata.nrows
#
#     humandatas = {}
#     for i in range(1, nrows):
#         if humandata.cell(i, 0).value == "":
#             continue
#         # print(humandata.cell(i,0).value)
#         qid = int(float(humandata.cell(i, 0).value))
#         score = humandata.cell(i, 1).value
#         gold = gold_answers[qid]
#         if gold != humandata.cell(i, 2).value:
#             print(gold)
#             print(humandata.cell(i, 2).value)
#         human_answers = [humandata.cell(i, 3).value, humandata.cell(i, 4).value, humandata.cell(i, 5).value]
#         humandatas[qid] = (score, gold, human_answers)
#
#
#     with open(mymethod, "r", encoding='utf-8') as f:
#         lines = f.readlines()
#         gsqa_answers = {}
#         for data, line in zip(test_datas, lines):
#             qid = data["id"]
#             gsqa_answers[qid] = line.strip()
#     with open(base_dir  + "/gsqa_test_human_result.txt", "w", encoding='utf-8') as f:
#         for q in humandatas.keys():
#             f.write(gsqa_answers[q] + '\n')
#     with open(base_dir  + "/gold_human_result.txt", "w", encoding='utf-8') as f:
#         for q in humandatas.keys():
#             f.write(gold_answers[q] + '\n')
#     methods_answers = {}
#
#
#
#     for root, dirs, files in os.walk(base_dir):
#         for dir in dirs:
#             print(base_dir+"/"+dir+"/test_result.txt")
#             with open(base_dir+"/"+dir+"/test_result.txt","r",encoding='utf-8') as f:
#                 lines=f.readlines()
#                 answers={}
#                 for data,line in zip(test_datas,lines):
#                     qid = data["id"]
#                     answers[qid]=line.strip()
#                 methods_answers[dir]=answers
#             with open(base_dir + "/" + dir + "/test_human_result.txt", "w", encoding='utf-8') as f:
#                 for q in humandatas.keys():
#                     f.write(answers[q]+'\n')
#
#     workbook = xlwt.Workbook(encoding='utf-8')
#     worksheet = workbook.add_sheet('answers')
#     worksheet.write(0, 0, "qid")
#     worksheet.write(0, 1, "score")
#     worksheet.write(0, 2, "gold")
#     worksheet.write(0, 3, "method")
#     worksheet.write(0, 4, "answers")
#
#
#
#     row = 1
#     for qid,(score,gold,human_answers) in humandatas.items():
#         baseline_answers = {method:answer[qid] for method,answer in methods_answers.items()}
#         gsqa_answer = gsqa_answers[qid]
#         worksheet.write_merge(row,row+len(human_answers)+len(baseline_answers) , 0, 0, qid)
#         worksheet.write_merge(row,row+len(human_answers)+len(baseline_answers) , 1, 1, score)
#         worksheet.write_merge(row,row+len(human_answers)+len(baseline_answers) , 2, 2, gold)
#         for idx,answer in enumerate(human_answers):
#             worksheet.write(row,3,"human"+str(idx))
#             worksheet.write(row,4,answer)
#             row+=1
#         worksheet.write(row,3,"gsqa")
#         worksheet.write(row,4,gsqa_answer)
#         row+=1
#         for method,answer in baseline_answers.items():
#             worksheet.write(row, 3, method)
#             worksheet.write(row, 4, answer)
#             row += 1
#     workbook.save(writefile)
#
# def random_method(readfile,methodfile,answerfile):
#     workbook_method = xlwt.Workbook(encoding='utf-8')
#     worksheet_method = workbook_method.add_sheet('method')
#     worksheet_method.write(0, 0, "qid")
#     worksheet_method.write(0, 1, "score")
#     worksheet_method.write(0, 2, "gold")
#     worksheet_method.write(0, 3, "method")
#
#     workbook_answer = xlwt.Workbook(encoding='utf-8')
#     worksheet_answer = workbook_answer.add_sheet('answers')
#     worksheet_answer.write(0, 0, "qid")
#     worksheet_answer.write(0, 1, "score")
#     worksheet_answer.write(0, 2, "gold")
#     worksheet_answer.write(0, 3, "answers")
#
#     humandata = xlrd.open_workbook(readfile)
#     humandata = humandata.sheet_by_index(0)
#     nrows = humandata.nrows
#
#     row = 1
#     data={}
#     for i in range(1, nrows):
#         if humandata.cell(i, 0).value == "":
#             data['answer'].append((humandata.cell(i,3).value,humandata.cell(i,4).value))
#         else:
#             if data:
#                 random.shuffle(data['answer'])
#                 worksheet_method.write_merge(row, row +  len(data['answer'])-1, 0, 0, data['qid'])
#                 worksheet_method.write_merge(row, row +  len(data['answer'])-1, 1, 1, data['score'])
#                 worksheet_method.write_merge(row, row +  len(data['answer'])-1, 2, 2, data['gold'])
#
#                 worksheet_answer.write_merge(row, row +  len(data['answer'])-1, 0, 0, data['qid'])
#                 worksheet_answer.write_merge(row, row +  len(data['answer'])-1, 1, 1, data['score'])
#                 worksheet_answer.write_merge(row, row  +  len(data['answer'])-1, 2, 2, data['gold'])
#
#                 for method, answer in data['answer']:
#                     worksheet_method.write(row, 3, method)
#                     if not str(method).startswith("human"):
#                         answer=answer[:evaluate.MAX_LEN]
#                     worksheet_answer.write(row, 3, answer)
#                     row += 1
#             qid = humandata.cell(i,0).value
#             score = humandata.cell(i,1).value
#             gold = humandata.cell(i,2).value
#             data["qid"]=qid
#             data["score"]=score
#             data["gold"]=gold
#             data['answer']=[(humandata.cell(i,3).value,humandata.cell(i,4).value)]
#     if data:
#         random.shuffle(data['answer'])
#         worksheet_method.write_merge(row, row + len(data['answer']) - 1, 0, 0, data['qid'])
#         worksheet_method.write_merge(row, row + len(data['answer']) - 1, 1, 1, data['score'])
#         worksheet_method.write_merge(row, row + len(data['answer']) - 1, 2, 2, data['gold'])
#
#         worksheet_answer.write_merge(row, row + len(data['answer']) - 1, 0, 0, data['qid'])
#         worksheet_answer.write_merge(row, row + len(data['answer']) - 1, 1, 1, data['score'])
#         worksheet_answer.write_merge(row, row + len(data['answer']) - 1, 2, 2, data['gold'])
#         for method, answer in data['answer']:
#             worksheet_method.write(row, 3, method)
#             if not method.startswith("human"):
#                 answer = answer[:evaluate.MAX_LEN]
#             worksheet_answer.write(row, 3, answer)
#             row += 1
#     workbook_answer.save(answerfile)
#     workbook_method.save(methodfile)
#
# def readloadscore(answer_file,method_file,joint_file,writefile):
#     joint_data = xlrd.open_workbook(joint_file)
#     joint_data = joint_data.sheet_by_index(0)
#     nrows = joint_data.nrows
#
#     answers = xlrd.open_workbook(answer_file)
#     answers = answers.sheet_by_index(0)
#
#     methods = xlrd.open_workbook(method_file)
#     methods = methods.sheet_by_index(0)
#
#     method_names = ["human0","human1","human2","gsqa","bert-nmt","BM25","DPR","HardEM",
#                "learning_to_retrieve","MASS","preSumm","smrs"]
#     scores={}
#     for method_name in method_names:
#         scores[method_name]=[]
#     workbook = xlwt.Workbook(encoding='utf-8')
#     worksheet = workbook.add_sheet('answers')
#     worksheet.write(0, 0, "qid")
#     worksheet.write(0, 1, "score")
#     worksheet.write(0, 2, "gold")
#     worksheet.write(0, 3, "method")
#     worksheet.write(0, 4, "answers")
#     worksheet.write(0, 5, "scores")
#
#     row = 1
#     data = {}
#     for i in range(1, nrows):
#         if answers.cell(i, 0).value == "":
#             data['answer'][methods.cell(i,3).value]=(answers.cell(i, 3).value, answers.cell(i, 4).value)
#         else:
#             if data:
#                 worksheet.write_merge(row, row + len(data['answer']) - 1, 0, 0, data['qid'])
#                 worksheet.write_merge(row, row + len(data['answer']) - 1, 1, 1, data['score'])
#                 worksheet.write_merge(row, row + len(data['answer']) - 1, 2, 2, data['gold'])
#
#                 for method_name in method_names:
#                     answer,score=data['answer'][method_name]
#                     scores[method_name].append(int(score))
#                     worksheet.write(row, 3, method_name)
#                     worksheet.write(row, 4, joint_data.cell(row,4).value)
#                     worksheet.write(row, 5, score)
#                     row += 1
#             qid = answers.cell(i, 0).value
#             score = answers.cell(i, 1).value
#             gold = answers.cell(i, 2).value
#             data["qid"] = qid
#             data["score"] = score
#             data["gold"] = gold
#             data['answer']={}
#             # print(answers.row(i))
#             # print(answers.cell(i,4).value)
#             data['answer'][methods.cell(i,3).value]=(answers.cell(i, 3).value, answers.cell(i, 4).value)
#     if data:
#         worksheet.write_merge(row, row + len(data['answer']) - 1, 0, 0, data['qid'])
#         worksheet.write_merge(row, row + len(data['answer']) - 1, 1, 1, data['score'])
#         worksheet.write_merge(row, row + len(data['answer']) - 1, 2, 2, data['gold'])
#
#         for method_name in method_names:
#             answer, score = data['answer'][method_name]
#             scores[method_name].append(int(score))
#             worksheet.write(row, 3, method_name)
#             worksheet.write(row, 4, joint_data.cell(row,4).value)
#             worksheet.write(row, 5, score)
#             row += 1
#
#     # workbook.save(writefile)
#     for method_name in method_names:
#         print(method_name,np.mean(scores[method_name]))
#
# def getHumanScore(readfile,writefile):
#     joint_data = xlrd.open_workbook(readfile)
#     joint_data = joint_data.sheet_by_index(0)
#     nrows = joint_data.nrows
#     data = {}
#     workbook = xlwt.Workbook(encoding='utf-8')
#     worksheet = workbook.add_sheet('answers')
#     worksheet.write(0, 0, "qid")
#     worksheet.write(0, 1, "score")
#     worksheet.write(0, 2, "gold")
#     worksheet.write(0, 3, "method")
#     worksheet.write(0, 4, "answers")
#     worksheet.write(0, 5, "scores")
#     row=1
#     method_names=["human0","human1","human2"]
#     for i in range(1, nrows):
#         if joint_data.cell(i, 0).value == "":
#             data['answer'][joint_data.cell(i, 3).value] = (joint_data.cell(i, 4).value, joint_data.cell(i, 5).value)
#         else:
#             if data:
#                 worksheet.write_merge(row, row + 2, 0, 0, data['qid'])
#                 worksheet.write_merge(row, row + 2, 1, 1, data['score'])
#                 worksheet.write_merge(row, row + 2, 2, 2, data['gold'])
#
#                 for method_name in method_names:
#                     answer, score = data['answer'][method_name]
#                     worksheet.write(row, 3, method_name)
#                     worksheet.write(row, 4, answer)
#                     worksheet.write(row, 5, round((score/10.0)*data['score']))
#                     row += 1
#             qid = joint_data.cell(i, 0).value
#             score = joint_data.cell(i, 1).value
#             gold = joint_data.cell(i, 2).value
#             data["qid"] = qid
#             data["score"] = score
#             data["gold"] = gold
#             data['answer'] = {}
#             # print(answers.row(i))
#             # print(answers.cell(i,4).value)
#             data['answer'][joint_data.cell(i, 3).value] = (joint_data.cell(i, 4).value, joint_data.cell(i, 5).value)
#     if data:
#         if data:
#             worksheet.write_merge(row, row + 2, 0, 0, data['qid'])
#             worksheet.write_merge(row, row + 2, 1, 1, data['score'])
#             worksheet.write_merge(row, row + 2, 2, 2, data['gold'])
#
#             for method_name in method_names:
#                 answer, score = data['answer'][method_name]
#                 worksheet.write(row, 3, method_name)
#                 worksheet.write(row, 4, answer)
#                 worksheet.write(row, 5, round((score / 10.0) * data['score']))
#                 row += 1
#
#     workbook.save(writefile)
#
# def transferJson2Tsv(readfiles,writefile):
#     write_datas=[]
#     for readfile in readfiles:
#         with open(readfile,'r',encoding='utf-8') as f:
#             datas=json.load(f)
#             for idx,data in enumerate(datas):
#                 question = data["question"]
#                 for p in data["pos_paragraphs"]:
#                     text_b = p
#                     label = "1"
#                     write_datas.append([str(idx),question.replace("\t","").replace("\n",""),text_b.replace("\t","").replace("\n",""),label])
#                 for p in data["neg_paragraphs"]:
#                     text_b = p
#                     label = "0"
#                     write_datas.append([str(idx), question.replace("\t","").replace("\n",""), text_b.replace("\t","").replace("\n",""), label])
#     with open(writefile,'w',encoding='utf-8') as f:
#         f.write("idx\ttext_a\ttext_b\tlabel\n")
#         for data in write_datas:
#             f.write('\t'.join(data)+'\n')
#
# def read_tsv(input_file, quotechar=None):
#     def csv_reader(fd, delimiter='\t'):
#         def gen():
#             for i in fd:
#                 slots = i.rstrip('\n').split(delimiter)
#                 if len(slots) == 1:
#                     yield slots,
#                 else:
#                     yield slots
#
#         return gen()
#     """Reads a tab separated value file."""
#     from collections import namedtuple
#     with open(input_file, 'r', encoding='utf8') as f:
#         reader = csv_reader(f)
#         headers = next(reader)
#         Example = namedtuple('Example', headers)
#
#         examples = []
#         for line in reader:
#             print(line)
#             example = Example(*line)
#             examples.append(example)
#         return examples

if __name__ == '__main__':
    # addRetrieveData('../data/train_raw.json', '../data/train_retrieve.json')
    # addRetrieveData('../data/val_raw.json', '../data/val_retrieve.json')
    # addRetrieveData('../data/test_raw.json', '../data/test_retrieve.json')
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

    generateGraphfile('../data/val_score.json',
                      '../data/val_graph.json')
    generateGraphfile('../data/test_score.json',
                      '../data/test_graph.json')
    generateGraphfile('../data/train_score.json',
                      '../data/train_graph.json')

    # generateGraphfileWithoutRank('../_data/test.json', '../_data/test_raw_graph.json')
    # generateGraphfileWithoutRank('../_data/val.json',
    #                              '../_data/val_raw_graph.json')
    # generateGraphfileWithoutRank('../_data/train.json',
    #                              '../_data/train_raw_graph.json')
    # with open("resource_open_domain/image_data_20201226/json_data/train.json",'r',encoding='utf-8') as f:
    #     datas = json.load(f)
    #     for data in datas:
    #         if data['id']==8562:
    #             print()
    #     print()
    # writeImageDataALLNew('resource_open_domain/image_data_20201227/json_data/image_data.json')
    # addRetrieveData('resource_open_domain/image_data_20201227/json_data/image_data.json', 'resource_open_domain/image_data_20201227/json_data/image_data_retrieve.json')
    # tagGoldInfo('resource_open_domain/image_data_20201227/json_data/image_data_retrieve.json','resource_open_domain/image_data_20201227/json_data/image_data_retrieve_gold.json')
    # splitTrainTest('resource_open_domain/image_data_20201227/json_data/image_data_retrieve_gold.json', 'resource_open_domain/image_data_20201227/json_data')
    # generateClassifyERNIEBysent('resource_open_domain/image_data_20201227/json_data/train.json','resource_open_domain/image_data_20201227/json_data/train.csv')
    # generateClassifyERNIEBysent('resource_open_domain/image_data_20201227/json_data/val.json','resource_open_domain/image_data_20201227/json_data/val.csv')
    # generateClassifyERNIEBysent('resource_open_domain/image_data_20201227/json_data/test.json','resource_open_domain/image_data_20201227/json_data/test.csv')
    # generateGraphfileWithoutRank('resource_open_domain/image_data_20201227/json_data/test.json', 'resource_open_domain/image_data_20201227/json_data/test_raw_graph.json')
    # generateGraphfileWithoutRank('resource_open_domain/image_data_20201227/json_data/val.json',
    #                              'resource_open_domain/image_data_20201227/json_data/val_raw_graph.json')
    # generateGraphfileWithoutRank('resource_open_domain/image_data_20201227/json_data/train.json',
    #                              'resource_open_domain/image_data_20201227/json_data/train_raw_graph.json')
    # import os
    # for root, ds, fs in os.walk('resource_open_domain/image_data_20201227/val_result'):
    #     for f in fs:
    #         step=f.split('.')[-1]
    #         print(f)
    #         calmetricERNIE('resource_open_domain/image_data_20201227/json_data/val.tsv', 'resource_open_domain/image_data_20201227/val_result/'+f, 'resource_open_domain/image_data_20201227/val_result_txt/'+step+'.txt')
    #         print()
    # calmetricERNIE('resource_open_domain/image_data_20201227/json_data/test.tsv', 'resource_open_domain/image_data_20201227/json_data/test_result.0.0', 'resource_open_domain/image_data_20201227/json_data/test.txt')
    # getanswer('resource_open_domain/image_data_20201227/json_data/test.json', 'resource_open_domain/image_data_20201227/json_data/test_gold.txt')

    # reloadERNIEscore('resource_open_domain/image_data_20201227/json_data/val.json',
    #                  'resource_open_domain/image_data_20201227/json_data/val.tsv',
    #                  'resource_open_domain/image_data_20201227/val_result/test_result.0.14394',
    #                  'resource_open_domain/image_data_20201227/json_data/val_score.json')
    # reloadERNIEscore('resource_open_domain/image_data_20201227/json_data/test.json',
    #                  'resource_open_domain/image_data_20201227/json_data/test.tsv',
    #                  'resource_open_domain/image_data_20201227/json_data/test_result.0.0',
    #                  'resource_open_domain/image_data_20201227/json_data/test_score.json')
    # generateGraphfile('resource_open_domain/image_data_20201227/json_data/val_score.json',
    #                   'resource_open_domain/image_data_20201227/json_data/val_graph.json')
    # generateGraphfile('resource_open_domain/image_data_20201227/json_data/test_score.json',
    #                   'resource_open_domain/image_data_20201227/json_data/test_graph.json')
    # reloadERNIEscore('resource_open_domain/image_data_20201227/json_data/train.json',
    #                  'resource_open_domain/image_data_20201227/json_data/train.tsv',
    #                  'resource_open_domain/image_data_20201227/json_data/train_result.0.0',
    #                  'resource_open_domain/image_data_20201227/json_data/train_score.json')
    # generateGraphfile('resource_open_domain/image_data_20201227/json_data/train_score.json',
    #                   'resource_open_domain/image_data_20201227/json_data/train_graph.json')
    # with open('../_data/test_graph.json','r',encoding='utf-8') as f:
    #     datas = json.load(f)
    #     print()
    # sampleForHuman('../_data/test.json',
    #                'resource_open_domain/image_data_20210118/human_performance')
    # sampleForHumanWithout('../_data/test.json',
    #                'resource_open_domain/image_data_20210118/human_performance_add',
    #                'resource_open_domain/image_data_20210118/human_performance_new/human_performance.json',
    #                )
    # get_SR3RES_humanperformance("resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate",
    #                             "resource_open_domain/image_data_20210118/human_performance_new/human_performance.json",
    #                             "../_data/test.json",
    #                             "resource_open_domain/image_data_20210118/human_performance_new/SR3_result.json")

    # change_qa_with_score("resource_open_domain/image_data_20210118/human_performance_new/human_performance.json",
    #                      "resource_open_domain/image_data_20210118/human_performance_new/human_performance_score.json")
    # combineBaselinesGSQAWithHuman("../_data/test.json",
    #                           "resource_open_domain/human_answers/考生答案.xlsx",
    #                           "resource_open_domain/image_data_20210118/baselines",
    #                           "resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate",
    #                           "resource_open_domain/human_answers/answer_human_gsqa_baselines.xls")
    # random_method("resource_open_domain/human_answers/answer_human_gsqa_baselines.xls",
    #               "resource_open_domain/human_answers/answer_human_gsqa_baselines_random_method.xls",
    #               "resource_open_domain/human_answers/answer_human_gsqa_baselines_random_answer.xls")
    # readloadscore("resource_open_domain/human_answers/answer_human_gsqa_baselines_random_answer.xls",
    #               "resource_open_domain/human_answers/answer_human_gsqa_baselines_random_method.xls",
    #               "resource_open_domain/human_answers/answer_human_gsqa_baselines.xls","resource_open_domain/human_answers/answer_human_gsqa_baselines_score.xls")
    # getHumanScore("resource_open_domain/human_answers/answer_human_gsqa_baselines_score.xls",
    #               "resource_open_domain/human_answers/human_scores.xls")
