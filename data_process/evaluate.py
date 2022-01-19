from rouge import Rouge
import numpy as np
import re
import jieba
# import xlrd
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
smooth = SmoothingFunction()
rouge = Rouge()
MAX_LEN = 60
stopwords = set([line.strip() for line in open('stopword', encoding='UTF-8')])
punctuation = [',', ';', '.', '，', '；', '。', '；', '？', '：', '、', '（', '）', '!', '！', '|']
punc = set(punctuation)

def removeunuseful(text):
    text = str(text)
    text = text.replace('\n', '')
    text = text.replace('\\n', '')
    text = text.replace('(', '（')
    text = text.replace(')', '）')
    text = text.replace('<q>', '')
    text = re.sub('\xa0+', '', text)
    text = re.sub('\u3000+', '', text)
    text = re.sub('\\s+', '', text)
    text = re.sub(' +','',text)
    score_p = '[（][^（）]*\d+[^（）]*分[^）]*[）]'
    text = re.sub(score_p, '', text)
    return  text

def removeStopwords(text,replace=False):
    text = str(text)
    text = text.replace('\n', '')
    text = text.replace('\\n', '')
    text = text.replace('(', '（')
    text = text.replace(')', '）')
    text = text.replace('<q>', '')
    text = re.sub('\xa0+', '', text)
    text = re.sub('\u3000+', '', text)
    text = re.sub('\\s+', '', text)
    score_p = '[（][^（）]*\d+[^（）]*分[^）]*[）]'
    text = re.sub(score_p, '', text)
    sentence_depart = jieba.cut(text.strip())
    outstr = []
    for word in sentence_depart:
        if word in punc:
            # pass
            if replace:
                outstr.append(punctuation[0])
        elif word not in stopwords:
            if word != '\t':
                outstr.append(word)
    text = ''.join(outstr)
    # punctuation_p = '[,;.，；。；？：、（）!！|]'
    # text = re.sub(punctuation_p, '', text)
    text = re.sub(' +', '', text)
    return text, outstr


memory = {}


def processText(rawText,replace=False):
    if rawText in memory:
        return memory[rawText]
    text, _ = removeStopwords(rawText,replace)
    text = ' '.join([w for w in text])
    memory[rawText] = text
    return text


def evalAnswer(answer, gold, metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f'),'BLEU'), max_len=MAX_LEN):
    zero_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    if type(answer) is str:
        if len(answer) == 0:
            scores = zero_scores
        else:
            answer = ' '.join(processText(answer,replace=True).split()[:max_len])
            gold = processText(gold,replace=True)
            if len(answer) == 0 or len(gold)==0:
                scores = zero_scores
            else:
                scores = rouge.get_scores(answer, gold)[0]
            if 'BLEU' in metrics:
                scores_bleu=sentence_bleu([gold], answer, smoothing_function=smooth.method1) #, weights=(0,0, 0,1)

        if metrics is None:
            return scores
        else:
            scores_list = []
            for i, m in enumerate(metrics):  # ('rouge-1','f')
                if m=='BLEU':
                    scores_list.append(scores_bleu)
                else:
                    scores_list.append(scores[m[0]][m[1]])
            return scores_list
    else:
        scores = [evalAnswer(a, g, metrics=metrics, max_len=max_len) for a, g in zip(answer, gold)]
        if metrics is None:
            return scores
        else:
            return np.mean(scores, axis=0).tolist()


def getScore(answerfile, goldfile):
    print(answerfile.split('/')[-2])
    # material_index=[]
    # material_index=[33, 90, 112, 134, 139, 175, 259, 305, 328, 333, 351, 399, 413, 429, 449, 521, 596, 651, 663, 674, 696, 725, 740, 748, 768, 858, 889, 894, 899, 931, 980, 1010, 1035, 1105, 1119, 1133, 1149, 1236, 1276, 1287, 1306, 1311, 1316, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1443, 1453, 1470, 1505, 1533, 1580, 1691, 1701, 1817, 1820, 1876, 1880, 1911, 2079, 2093, 2102, 2140, 2297, 2355, 2363, 2422, 2489, 2525, 2565, 2572, 2584, 2613]
    # material_index=[11, 14, 33, 90, 112, 134, 139, 175, 191, 212, 238, 259, 305, 328, 333, 351, 399, 400, 413, 429, 449, 494, 521, 596, 651, 663, 674, 687, 696, 710, 725, 740, 748, 768, 783, 801, 858, 889, 894, 899, 931, 980, 1010, 1035, 1045, 1088, 1105, 1119, 1133, 1147, 1149, 1236, 1276, 1287, 1300, 1306, 1311, 1316, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1443, 1453, 1470, 1505, 1533, 1580, 1638, 1691, 1701, 1817, 1820, 1876, 1880, 1898, 1911, 2017, 2079, 2090, 2093, 2102, 2140, 2152, 2189, 2297, 2355, 2363, 2372, 2422, 2489, 2525, 2565, 2572, 2584, 2613, 2617, 2665]
    # material_index=[8, 11, 73, 75, 76, 90, 95, 97, 127, 134, 161, 168, 169, 237, 259, 261, 280, 326, 328, 349, 350, 399, 413, 421, 429, 479, 501, 521, 539, 576, 580, 597, 618, 651, 663, 664, 674, 710, 725, 727, 740, 748, 768, 783, 801, 889, 894, 898, 899, 931, 940, 959, 979, 980, 988, 1010, 1133, 1142, 1149, 1167, 1213, 1239, 1245, 1268, 1277, 1285, 1289, 1297, 1306, 1316, 1331, 1355, 1370, 1377, 1384, 1418, 1420, 1423, 1437, 1441, 1443, 1479, 1533, 1580, 1638, 1659, 1660, 1675, 1701, 1716, 1774, 1814, 1817, 1819, 1876, 1880, 1892, 1903, 1911, 1997, 2021, 2043, 2079, 2092, 2093, 2102, 2120, 2140, 2173, 2180, 2189, 2209, 2246, 2326, 2343, 2363, 2422, 2444, 2495, 2525, 2565, 2574, 2584, 2613, 2623, 2671, 2678]
    # material_index=[90, 112, 134, 175, 259, 305, 328, 399, 413, 429, 521, 596, 651, 663, 674, 696, 725, 740, 748, 768, 858, 889, 894, 899, 980, 1010, 1119, 1133, 1149, 1236, 1287, 1306, 1311, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1533, 1580, 1701, 1817, 1820, 1876, 1880, 1911, 2079, 2093, 2102, 2140, 2355, 2363, 2422, 2525, 2565, 2584, 2613]
    # material_index=[8, 11, 30, 53, 73, 75, 76, 90, 95, 97, 127, 134, 161, 168, 169, 230, 237, 245, 259, 261, 280, 319, 326, 328, 339, 349, 350, 351, 395, 399, 413, 421, 429, 479, 486, 493, 501, 521, 539, 566, 576, 580, 596, 597, 618, 651, 663, 664, 674, 710, 725, 727, 728, 740, 748, 768, 783, 801, 889, 894, 898, 899, 931, 940, 959, 979, 980, 988, 1010, 1133, 1142, 1149, 1167, 1213, 1218, 1239, 1245, 1268, 1277, 1285, 1289, 1297, 1300, 1306, 1316, 1331, 1355, 1370, 1377, 1384, 1395, 1418, 1420, 1423, 1437, 1441, 1443, 1453, 1479, 1505, 1533, 1580, 1599, 1600, 1638, 1659, 1660, 1675, 1701, 1716, 1774, 1807, 1814, 1817, 1819, 1854, 1876, 1880, 1892, 1903, 1911, 1997, 2021, 2043, 2079, 2092, 2093, 2095, 2102, 2120, 2140, 2173, 2180, 2189, 2209, 2246, 2259, 2306, 2326, 2335, 2343, 2355, 2362, 2363, 2422, 2444, 2495, 2499, 2516, 2525, 2565, 2574, 2584, 2605, 2613, 2623, 2671, 2678]
    # do_eval=[]
    # with open('resource/tagDataRaw/test_info','r',encoding='utf-8') as f:
    #     for line in f:
    #         search,qtype=line.strip().split('\t')
    #         if search == 'False' and qtype=='generate':
    #             do_eval.append(True)
    #         else:
    #             do_eval.append(False)
    # need_material=[]
    # with open('resource/tagDataKey/abs_bert_geo.1000.raw_src','r',encoding='utf-8') as f:
    #     for line in f:
    #         if '[SEP][CLS]' in line:
    #             need_material.append(True)
    #         else:need_material.append(False)
    metrics = [
        # ("rouge-1", 'r'),
        # ("rouge-1", 'p'),
        ("rouge-1", 'f'),
        # ("rouge-2", 'r'),
        # ("rouge-2", 'p'),
        ("rouge-2", 'f'),
        # ("rouge-l", 'r'),
        # ("rouge-l", 'p'),
        ("rouge-l", 'f'),
        'BLEU'
    ]
    scoreAll = [[] for _ in range(len(metrics))]
    with open(answerfile, "r", encoding='utf-8') as f_answer:
        answers = f_answer.readlines()
        with open(goldfile, "r", encoding='utf-8') as f_gold:
            golds = f_gold.readlines()
            # print()
            for idx, (answer, gold) in enumerate(zip(answers, golds)):
                # if not do_eval[idx] or not need_material[idx]:
                #     continue
                # if idx not in material_index:
                #     continue
                answer = answer.strip()
                gold = gold.strip()
                # print(answer)
                # print(gold)
                if len(answer) != 0:
                    try:
                        score = evalAnswer(answer, gold, metrics=metrics)
                        # if score[5]>0.13:
                        #     material_index.append(idx)
                        # print(score)
                        for i in range(len(metrics)):
                            scoreAll[i].append(score[i]*100)
                    except:
                        for i in range(len(scoreAll)):
                            scoreAll[i].append(0)
                else:
                    for i in range(len(scoreAll)):
                        scoreAll[i].append(0)
    # print("nan score")
    # scoreAvg = np.nanmean(scoreAll, axis=1)
    # for metric, score in zip(metrics, scoreAvg):
    #     print(metric, score)
    print("score")
    # print(material_index)
    # print(len(material_index))
    scoreAvg = np.mean(scoreAll, axis=1)
    for metric, score in zip(metrics, scoreAvg):
        print(metric, round(score,2))
    return scoreAll

def getScoreForHuman(readfile):
    # material_index=[]
    # material_index=[33, 90, 112, 134, 139, 175, 259, 305, 328, 333, 351, 399, 413, 429, 449, 521, 596, 651, 663, 674, 696, 725, 740, 748, 768, 858, 889, 894, 899, 931, 980, 1010, 1035, 1105, 1119, 1133, 1149, 1236, 1276, 1287, 1306, 1311, 1316, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1443, 1453, 1470, 1505, 1533, 1580, 1691, 1701, 1817, 1820, 1876, 1880, 1911, 2079, 2093, 2102, 2140, 2297, 2355, 2363, 2422, 2489, 2525, 2565, 2572, 2584, 2613]
    # material_index=[11, 14, 33, 90, 112, 134, 139, 175, 191, 212, 238, 259, 305, 328, 333, 351, 399, 400, 413, 429, 449, 494, 521, 596, 651, 663, 674, 687, 696, 710, 725, 740, 748, 768, 783, 801, 858, 889, 894, 899, 931, 980, 1010, 1035, 1045, 1088, 1105, 1119, 1133, 1147, 1149, 1236, 1276, 1287, 1300, 1306, 1311, 1316, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1443, 1453, 1470, 1505, 1533, 1580, 1638, 1691, 1701, 1817, 1820, 1876, 1880, 1898, 1911, 2017, 2079, 2090, 2093, 2102, 2140, 2152, 2189, 2297, 2355, 2363, 2372, 2422, 2489, 2525, 2565, 2572, 2584, 2613, 2617, 2665]
    # material_index=[8, 11, 73, 75, 76, 90, 95, 97, 127, 134, 161, 168, 169, 237, 259, 261, 280, 326, 328, 349, 350, 399, 413, 421, 429, 479, 501, 521, 539, 576, 580, 597, 618, 651, 663, 664, 674, 710, 725, 727, 740, 748, 768, 783, 801, 889, 894, 898, 899, 931, 940, 959, 979, 980, 988, 1010, 1133, 1142, 1149, 1167, 1213, 1239, 1245, 1268, 1277, 1285, 1289, 1297, 1306, 1316, 1331, 1355, 1370, 1377, 1384, 1418, 1420, 1423, 1437, 1441, 1443, 1479, 1533, 1580, 1638, 1659, 1660, 1675, 1701, 1716, 1774, 1814, 1817, 1819, 1876, 1880, 1892, 1903, 1911, 1997, 2021, 2043, 2079, 2092, 2093, 2102, 2120, 2140, 2173, 2180, 2189, 2209, 2246, 2326, 2343, 2363, 2422, 2444, 2495, 2525, 2565, 2574, 2584, 2613, 2623, 2671, 2678]
    # material_index=[90, 112, 134, 175, 259, 305, 328, 399, 413, 429, 521, 596, 651, 663, 674, 696, 725, 740, 748, 768, 858, 889, 894, 899, 980, 1010, 1119, 1133, 1149, 1236, 1287, 1306, 1311, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1533, 1580, 1701, 1817, 1820, 1876, 1880, 1911, 2079, 2093, 2102, 2140, 2355, 2363, 2422, 2525, 2565, 2584, 2613]
    # material_index=[8, 11, 30, 53, 73, 75, 76, 90, 95, 97, 127, 134, 161, 168, 169, 230, 237, 245, 259, 261, 280, 319, 326, 328, 339, 349, 350, 351, 395, 399, 413, 421, 429, 479, 486, 493, 501, 521, 539, 566, 576, 580, 596, 597, 618, 651, 663, 664, 674, 710, 725, 727, 728, 740, 748, 768, 783, 801, 889, 894, 898, 899, 931, 940, 959, 979, 980, 988, 1010, 1133, 1142, 1149, 1167, 1213, 1218, 1239, 1245, 1268, 1277, 1285, 1289, 1297, 1300, 1306, 1316, 1331, 1355, 1370, 1377, 1384, 1395, 1418, 1420, 1423, 1437, 1441, 1443, 1453, 1479, 1505, 1533, 1580, 1599, 1600, 1638, 1659, 1660, 1675, 1701, 1716, 1774, 1807, 1814, 1817, 1819, 1854, 1876, 1880, 1892, 1903, 1911, 1997, 2021, 2043, 2079, 2092, 2093, 2095, 2102, 2120, 2140, 2173, 2180, 2189, 2209, 2246, 2259, 2306, 2326, 2335, 2343, 2355, 2362, 2363, 2422, 2444, 2495, 2499, 2516, 2525, 2565, 2574, 2584, 2605, 2613, 2623, 2671, 2678]
    # do_eval=[]
    # with open('resource/tagDataRaw/test_info','r',encoding='utf-8') as f:
    #     for line in f:
    #         search,qtype=line.strip().split('\t')
    #         if search == 'False' and qtype=='generate':
    #             do_eval.append(True)
    #         else:
    #             do_eval.append(False)
    # need_material=[]
    # with open('resource/tagDataKey/abs_bert_geo.1000.raw_src','r',encoding='utf-8') as f:
    #     for line in f:
    #         if '[SEP][CLS]' in line:
    #             need_material.append(True)
    #         else:need_material.append(False)
    metrics = [
        # ("rouge-1", 'r'),
        # ("rouge-1", 'p'),
        ("rouge-1", 'f'),
        # ("rouge-2", 'r'),
        # ("rouge-2", 'p'),
        ("rouge-2", 'f'),
        # ("rouge-l", 'r'),
        # ("rouge-l", 'p'),
        ("rouge-l", 'f')]
    humandata = xlrd.open_workbook(readfile)
    humandata = humandata.sheet_by_index(0)
    nrows = humandata.nrows
    data = {}
    datas =[]
    for i in range(1, nrows):
        # temp=humandata.cell(i,0).value
        if humandata.cell(i, 0).value == "":
            data['answer'][humandata.cell(i, 3).value]= humandata.cell(i, 4).value
        else:
            if data:
                datas.append(data)
            data={}
            qid = humandata.cell(i, 0).value
            score = humandata.cell(i, 1).value
            gold = humandata.cell(i, 2).value
            data["qid"] = qid
            data["score"] = score
            data["gold"] = gold
            data['answer'] = {humandata.cell(i, 3).value: humandata.cell(i, 4).value}
    if data:
        datas.append(data)
    # print(datas[0]['answer'].keys())
    for method in datas[0]['answer'].keys():
        print(method+":")
        answers=[data['answer'][method]  for data in datas]
        golds = [data['gold']  for data in datas]
        # print()
        scoreAll = [[] for _ in range(len(metrics))]
        for idx, (answer, gold) in enumerate(zip(answers, golds)):
            # if not do_eval[idx] or not need_material[idx]:
            #     continue
            # if idx not in material_index:
            #     continue
            answer = answer.strip()
            gold = gold.strip()
            # print(answer)
            # print(gold)
            if len(answer) != 0:
                try:
                    score = evalAnswer(answer, gold, metrics=metrics)
                    # if score[5]>0.13:
                    #     material_index.append(idx)
                    # print(score)
                    for i in range(len(metrics)):
                        scoreAll[i].append(score[i]*100)
                except:
                    for i in range(len(scoreAll)):
                        scoreAll[i].append(0)
            else:
                for i in range(len(scoreAll)):
                    scoreAll[i].append(0)
        print("score")
        scoreAvg = np.mean(scoreAll, axis=1)
        for metric, score in zip(metrics, scoreAvg):
            print(metric, score)


def cmpScore(answerfile1,answerfile2, goldfile):

    metrics = [
        # ("rouge-1", 'r'),
        # ("rouge-1", 'p'),
        ("rouge-1", 'f'),
        # ("rouge-2", 'r'),
        # ("rouge-2", 'p'),
        ("rouge-2", 'f'),
        # ("rouge-l", 'r'),
        # ("rouge-l", 'p'),
        ("rouge-l", 'f')]
    scoreAll = [[] for _ in range(len(metrics))]
    with open(answerfile1, "r", encoding='utf-8') as f_answer1:
        answers1 = f_answer1.readlines()
        with open(answerfile2, "r", encoding='utf-8') as f_answer2:
            answers2 = f_answer2.readlines()
        with open(goldfile, "r", encoding='utf-8') as f_gold:
            golds = f_gold.readlines()
            for idx, (answer1,answer2, gold) in enumerate(zip(answers1,answers2, golds)):
                # if not do_eval[idx] or not need_material[idx]:
                #     continue
                # if idx not in material_index:
                #     continue
                answer1 = answer1.strip()
                answer2 = answer2.strip()
                gold = gold.strip()

                if len(answer1) != 0 and len(answer2)!=0:
                    try:
                        score1 = evalAnswer(answer1, gold, metrics=metrics)
                        score2 = evalAnswer(answer2, gold, metrics=metrics)
                        if(score1[2]<score2[2] and score1[1]>score2[1]):
                            print(answer1)
                            print(answer2)
                            print(gold)
                            print(score1,score2)
                        # if score[5]>0.13:
                        #     material_index.append(idx)

                    except:
                        for i in range(len(scoreAll)):
                            scoreAll[i].append(0)



import json
def getScoreForHardEM(answerfile, goldfile):
    metrics = [
        ("rouge-1", 'r'),
        ("rouge-1", 'p'),
        ("rouge-1", 'f'),
        ("rouge-2", 'r'),
        ("rouge-2", 'p'),
        ("rouge-2", 'f'),
        ("rouge-l", 'r'),
        ("rouge-l", 'p'),
        ("rouge-l", 'f')]
    scoreAll = [[] for _ in range(len(metrics))]
    with open(answerfile, "r", encoding='utf-8') as f_answer:
        data = json.load(f_answer)
        keys=data.keys()
        with open(goldfile, "r", encoding='utf-8') as f_gold:
            golds = f_gold.readlines()
            for idx, (key, gold) in enumerate(zip(keys, golds)):
                # print(idx)
                answer_list = [item["text"] for item in data[key]]
                gold = gold.strip()
                # print(answer)
                max_score=-1
                max_score_all=[]
                for answer in answer_list:
                    answer=answer.replace(" ","")
                    if len(answer) != 0:
                        try:
                            score = evalAnswer(answer[:MAX_LEN], gold)
                            if score[2]>max_score or max_score==-1:
                                max_score=score[2]
                                for i in range(len(metrics)):
                                    max_score_all.append(score[i])
                        except:
                            for i in range(len(scoreAll)):
                                max_score_all.append(0)
                    else:
                        for i in range(len(scoreAll)):
                            scoreAll[i].append(0)
                for i in range(len(scoreAll)):
                    scoreAll[i].append(max_score_all[i])


    print("score")
    scoreAvg = np.mean(scoreAll, axis=1)
    for metric, score in zip(metrics, scoreAvg):
        print(metric, score)
    return scoreAvg

def average_precision(gt, pred,maxnum=-1):
  """
  Computes the average precision.

  This function computes the average prescision at k between two lists of
  items.

  Parameters
  ----------
  gt: set
       A set of ground-truth elements (order doesn't matter)
  pred: list
        A list of predicted elements (order does matter)

  Returns
  -------
  score: double
      The average precision over the input lists
  """

  if not gt:
    return 0.0

  score = 0.0
  num_hits = 0.0
  for i,p in enumerate(pred):
        if maxnum!=-1 and i >=maxnum:
            break
        if p in gt and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
  if maxnum==-1:
     maxnum=len(gt)
  return score / max(1.0, min(len(gt),maxnum))

def hit(gt,pred,maxnum=-1):
    num_hits = 0.0
    for i, p in enumerate(pred):
        if maxnum!=-1 and i >=maxnum:
            break
        if p in gt and p not in pred[:i]:
            num_hits+=1.0
    if maxnum==-1:
        maxnum=len(gt)
    return num_hits/max(1.0, min(len(gt),maxnum))


def NDCG(gt, pred,maxnum=-1, use_graded_scores=False):
  score = 0.0
  for rank, item in enumerate(pred):
    if maxnum!=-1 and rank>=maxnum:
        break
    if item in gt:
      if use_graded_scores:
        grade = 1.0 / (gt.index(item) + 1)
      else:
        grade = 1.0
      score += grade / np.log2(rank + 2)

  norm = 0.0
  for rank in range(len(gt)):
    if maxnum!=-1 and rank>=maxnum:
        break
    if use_graded_scores:
      grade = 1.0 / (rank + 1)
    else:
      grade = 1.0
    norm += grade / np.log2(rank + 2)
  return score / max(0.3, norm)


def cal_metrics(gt, pred, metrics_map,maxnum=-1):
  '''
  Returns a numpy array containing metrics specified by metrics_map.
  gt: set
      A set of ground-truth elements (order doesn't matter)
  pred: list
      A list of predicted elements (order does matter)
  '''
  out = np.zeros((len(metrics_map),), np.float32)

  if ('MAP' in metrics_map):
    avg_precision = average_precision(gt=gt, pred=pred,maxnum=maxnum)
    out[metrics_map.index('MAP')] = avg_precision

  if ('RPrec' in metrics_map):
    intersec = len(gt & set(pred[:len(gt)]))
    out[metrics_map.index('RPrec')] = intersec / max(1., float(len(gt)))

  if 'MRR' in metrics_map:
    score = 0.0
    for rank, item in enumerate(pred):
        if maxnum != -1 and rank >= maxnum:
            break
        if item in gt:
            score = 1.0 / (rank + 1.0)
            break
    out[metrics_map.index('MRR')] = score

  if 'MRR@10' in metrics_map:
    score = 0.0
    for rank, item in enumerate(pred[:10]):
      if item in gt:
        score = 1.0 / (rank + 1.0)
        break
    out[metrics_map.index('MRR@10')] = score

  if ('NDCG' in metrics_map):
    out[metrics_map.index('NDCG')] = NDCG(gt, pred,maxnum=maxnum)

  if ('HIT' in metrics_map):
    out[metrics_map.index('HIT')] = hit(gt, pred,maxnum=maxnum)
  return out

import scipy.stats as stats
def ttest(scoreAll1,scoreAll2):
    # print(stats.levene(scoreAll1, scoreAll2))
    # print(stats.ttest_ind(scoreAll1, scoreAll2))
    print(stats.ttest_rel(scoreAll1, scoreAll2))

def ttestPair(readfile1,readfile2,goldfile):
    scoreAll1=getScore(readfile1, goldfile)
    scoreAll2=getScore(readfile2, goldfile)
    print(len(scoreAll1))
    ttest(scoreAll1[0],scoreAll2[0])
    ttest(scoreAll1[1], scoreAll2[1])
    ttest(scoreAll1[2], scoreAll2[2])
    ttest(scoreAll1[3],scoreAll2[3])

def eval_xunfei(readfile,writefile):
    import csv
    # import pandas as pd
    # data = pd.read_csv(readfile)
    # print(data)
    with open(readfile, 'r',encoding='gbk') as f:
        with open(writefile, 'w', encoding='utf-8',newline='') as f_w:
            f.readline()
            reader = csv.reader(f)
            writer = csv.writer(f_w)
            writer.writerow(["试卷","场景","问题","答案","系统答案","人工评分","rouge-1-f","rouge-2-f",'rouge-l-f',"rouge-1-r","rouge-2-r",'rouge-l-r'])
            for row in reader:
                row[-1]=float(row[-1])
                gold=row[3]
                candidate=row[4]
                score = evalAnswer(candidate, gold, metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f'),('rouge-1', 'r'), ('rouge-2', 'r'), ('rouge-l', 'r')), max_len=10000)
                row.extend(score)
                # print(row)
                writer.writerow(row)
                # assert len(row)==9

def cal_pearson_spearman(readfile):
    import pandas as pd
    import matplotlib.pyplot as plt
    datas = pd.read_csv(readfile,encoding='utf-8')
    rouge1f = datas[['人工评分','rouge-1-f']]
    temp=rouge1f.corr()
    print("rouge1f: pearson: ",rouge1f.corr().iat[0,1] )#计算皮尔逊相关系数
    print("rouge1f: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

    rouge1f = datas[['人工评分','rouge-2-f']]
    print("rouge2f: pearson: ",rouge1f.corr().iat[0,1])#计算皮尔逊相关系数
    print("rouge2f: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

    rouge1f = datas[['人工评分','rouge-l-f']]
    print("rougelf: pearson: ",rouge1f.corr().iat[0,1])#计算皮尔逊相关系数
    print("rougelf: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

    rouge1f = datas[['人工评分','rouge-1-r']]
    print("rouge1r: pearson: ",rouge1f.corr().iat[0,1])#计算皮尔逊相关系数
    print("rouge1r: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

    rouge1f = datas[['人工评分','rouge-2-r']]
    print("rouge2r: pearson: ",rouge1f.corr().iat[0,1])#计算皮尔逊相关系数
    print("rouge2r: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

    rouge1f = datas[['人工评分','rouge-l-r']]
    print("rougelr: pearson: ",rouge1f.corr().iat[0,1])#计算皮尔逊相关系数
    print("rougelr: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

    rouge1f = datas[['人工评分','bert_dual']]
    print("bert_dual: pearson: ",rouge1f.corr().iat[0,1])#计算皮尔逊相关系数
    print("bert_dual: spearman: ",rouge1f.corr('spearman').iat[0,1])#计算spearman相关系数

if __name__ == '__main__':
    getScore('resource_open_domain/image_data_20210118/baselines/MASS/dev_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/MASS/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/preSumm/dev.19500.candidate',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/preSumm/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/bert-nmt/dev.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/bert-nmt/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/json_data/val.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/json_data/test.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/BM25/val_rt_answer.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/BM25/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/smrs/dev_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/smrs/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/HardEM/dev_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/HardEM/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/learning_to_retrieve/dev.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/learning_to_retrieve/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/DPR/dev.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    getScore('resource_open_domain/image_data_20210118/baselines/DPR/test_result.txt',
             'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')

    # getScore('resource_open_domain/ckgg_data/ernie_test.txt',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')


    # getScore('resource_open_domain/image_data_20210118/results_multidoc/.0.candidate',
    #          'resource_open_domain/image_data_20210118/results_multidoc/.0.gold')


    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent/.16500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copyword_graph/.14500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copyword/.20000.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_multidoc/.19500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_noScenario/.17000.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_onlyScenario/.18500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copyword_graph/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copyword/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_multidoc/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_noScenario/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_onlyScenario/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')

    # cmpScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate','resource_open_domain/image_data_20210118/my_results/results_multidoc/.19500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_0.2/.15500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_0.4/.18500.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_5/.14000.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_10/.19000.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')

    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_0.2/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_0.4/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_5/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # getScore('resource_open_domain/image_data_20210118/my_results/results_copysent_graph_10/.0.candidate',
    #          'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')

    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copyword/.20000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copyword_graph/.14500.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copysent/.16500.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_multidoc/.19500.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_onlyScenario/.18500.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copysent_graph_noScenario/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')

    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copyword/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copyword_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copysent/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_multidoc/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_onlyScenario/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/my_results/results_copysent_graph_noScenario/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')

    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val_rt_answer.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/smrs/dev_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/MASS/generate_15_dev.out',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/preSumm/dev.19500.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/HardEM/dev_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/json_data/val.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/DPR/dev.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/learning_to_retrieve/dev.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.17000.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/bert-nmt/dev.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/val.tgt')
    #
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/smrs/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/MASS/generate_15_test.out',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/preSumm/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/HardEM/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/json_data/test.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/DPR/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/learning_to_retrieve/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')
    # ttestPair('resource_open_domain/image_data_20210118/my_results/results_copysent_graph/.0.candidate',
    #           'resource_open_domain/image_data_20210118/baselines/bert-nmt/test_result.txt',
    #           'resource_open_domain/image_data_20210118/baselines/BM25/test.tgt')