from rouge import Rouge
import numpy as np
import re
import jieba
import traceback
rouge = Rouge()
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
MAX_LEN = 60
# MAX_LEN = 400
stopwords_zh = set([line.strip() for line in open('stopword', encoding='UTF-8')])
punctuation = [',', ';', '.', '，', '；', '。', '；', '？', '：', '、', '（', '）', '!', '！', '|']
punc = set(punctuation)

stop_words_en = set(stopwords.words('english'))
punctuation_en = [",", ";", ".",'?','!','...']
def processText_en(example_sent):
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [w for w in word_tokens if not w in stop_words_en and not  w in punctuation_en]
    return ' '.join(filtered_sentence)

def removeStopwords(text):
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
            outstr.append(punctuation[0])
        elif word not in stopwords_zh:
            if word != '\t':
                outstr.append(word)
    text = ''.join(outstr)
    # punctuation_p = '[,;.，；。；？：、（）!！|]'
    # text = re.sub(punctuation_p, '', text)
    text = re.sub(' +', '', text)
    return text, outstr


memory = {}
def processText(rawText):
    if rawText in memory:
        return memory[rawText]
    text, _ = removeStopwords(rawText)
    text = ' '.join([w for w in text])
    memory[rawText] = text
    return text


def evalAnswer(answer, gold, metrics=(('rouge-1', 'f'), ('rouge-2', 'f'), ('rouge-l', 'f')), max_len=MAX_LEN,language='zh'):
    zero_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                   'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    if type(answer) is str:
        if len(answer) == 0:
            scores = zero_scores
        else:
            if language=='zh':
                answer = ' '.join(processText(answer).split()[:max_len])
                gold = processText(gold)
            else:
                answer = ' '.join(processText_en(answer).split()[:max_len])
                gold = processText_en(gold)
            if len(answer) == 0 or len(gold)==0:
                scores = zero_scores
            else:
                # print('a:',answer)
                # print('g:',gold)
                scores = rouge.get_scores(answer, gold)[0]
        if metrics is None:
            return scores
        else:
            scores_list = []
            for i, m in enumerate(metrics):  # ('rouge-1','f')
                scores_list.append(scores[m[0]][m[1]])
            return scores_list
    else:
        scores = [evalAnswer(a, g, metrics=metrics, max_len=max_len,language=language) for a, g in zip(answer, gold)]
        if metrics is None:
            return scores
        else:
            return np.mean(scores, axis=0).tolist()


def getScore(answerfile, goldfile,language):
    # material_index=[]
    # material_index=[33, 90, 112, 134, 139, 175, 259, 305, 328, 333, 351, 399, 413, 429, 449, 521, 596, 651, 663, 674, 696, 725, 740, 748, 768, 858, 889, 894, 899, 931, 980, 1010, 1035, 1105, 1119, 1133, 1149, 1236, 1276, 1287, 1306, 1311, 1316, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1443, 1453, 1470, 1505, 1533, 1580, 1691, 1701, 1817, 1820, 1876, 1880, 1911, 2079, 2093, 2102, 2140, 2297, 2355, 2363, 2422, 2489, 2525, 2565, 2572, 2584, 2613]
    # material_index=[11, 14, 33, 90, 112, 134, 139, 175, 191, 212, 238, 259, 305, 328, 333, 351, 399, 400, 413, 429, 449, 494, 521, 596, 651, 663, 674, 687, 696, 710, 725, 740, 748, 768, 783, 801, 858, 889, 894, 899, 931, 980, 1010, 1035, 1045, 1088, 1105, 1119, 1133, 1147, 1149, 1236, 1276, 1287, 1300, 1306, 1311, 1316, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1443, 1453, 1470, 1505, 1533, 1580, 1638, 1691, 1701, 1817, 1820, 1876, 1880, 1898, 1911, 2017, 2079, 2090, 2093, 2102, 2140, 2152, 2189, 2297, 2355, 2363, 2372, 2422, 2489, 2525, 2565, 2572, 2584, 2613, 2617, 2665]
    # material_index=[8, 11, 73, 75, 76, 90, 95, 97, 127, 134, 161, 168, 169, 237, 259, 261, 280, 326, 328, 349, 350, 399, 413, 421, 429, 479, 501, 521, 539, 576, 580, 597, 618, 651, 663, 664, 674, 710, 725, 727, 740, 748, 768, 783, 801, 889, 894, 898, 899, 931, 940, 959, 979, 980, 988, 1010, 1133, 1142, 1149, 1167, 1213, 1239, 1245, 1268, 1277, 1285, 1289, 1297, 1306, 1316, 1331, 1355, 1370, 1377, 1384, 1418, 1420, 1423, 1437, 1441, 1443, 1479, 1533, 1580, 1638, 1659, 1660, 1675, 1701, 1716, 1774, 1814, 1817, 1819, 1876, 1880, 1892, 1903, 1911, 1997, 2021, 2043, 2079, 2092, 2093, 2102, 2120, 2140, 2173, 2180, 2189, 2209, 2246, 2326, 2343, 2363, 2422, 2444, 2495, 2525, 2565, 2574, 2584, 2613, 2623, 2671, 2678]
    # material_index=[90, 112, 134, 175, 259, 305, 328, 399, 413, 429, 521, 596, 651, 663, 674, 696, 725, 740, 748, 768, 858, 889, 894, 899, 980, 1010, 1119, 1133, 1149, 1236, 1287, 1306, 1311, 1329, 1331, 1385, 1420, 1423, 1437, 1441, 1533, 1580, 1701, 1817, 1820, 1876, 1880, 1911, 2079, 2093, 2102, 2140, 2355, 2363, 2422, 2525, 2565, 2584, 2613]
    # material_index=[8, 11, 30, 53, 73, 75, 76, 90, 95, 97, 127, 134, 161, 168, 169, 230, 237, 245, 259, 261, 280, 319, 326, 328, 339, 349, 350, 351, 395, 399, 413, 421, 429, 479, 486, 493, 501, 521, 539, 566, 576, 580, 596, 597, 618, 651, 663, 664, 674, 710, 725, 727, 728, 740, 748, 768, 783, 801, 889, 894, 898, 899, 931, 940, 959, 979, 980, 988, 1010, 1133, 1142, 1149, 1167, 1213, 1218, 1239, 1245, 1268, 1277, 1285, 1289, 1297, 1300, 1306, 1316, 1331, 1355, 1370, 1377, 1384, 1395, 1418, 1420, 1423, 1437, 1441, 1443, 1453, 1479, 1505, 1533, 1580, 1599, 1600, 1638, 1659, 1660, 1675, 1701, 1716, 1774, 1807, 1814, 1817, 1819, 1854, 1876, 1880, 1892, 1903, 1911, 1997, 2021, 2043, 2079, 2092, 2093, 2095, 2102, 2120, 2140, 2173, 2180, 2189, 2209, 2246, 2259, 2306, 2326, 2335, 2343, 2355, 2362, 2363, 2422, 2444, 2495, 2499, 2516, 2525, 2565, 2574, 2584, 2605, 2613, 2623, 2671, 2678]

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
        answers = f_answer.readlines()
        with open(goldfile, "r", encoding='utf-8') as f_gold:
            golds = f_gold.readlines()
            for idx, (answer, gold) in enumerate(zip(answers, golds)):
                # if idx not in material_index:
                #     continue
                answer = answer.strip()
                gold = gold.strip()
                # print(answer)
                # print(gold)
                if len(answer) != 0:
                    try:
                        score = evalAnswer(answer, gold, metrics=metrics,language=language)
                        # if score[5]>0.13:
                        #     material_index.append(idx)
                        # print(score)
                        for i in range(len(metrics)):
                            scoreAll[i].append(score[i])
                    except:
                        traceback.print_exc()
                        for i in range(len(scoreAll)):
                            scoreAll[i].append(0)
                else:
                    for i in range(len(scoreAll)):
                        scoreAll[i].append(0)
    # print("nan score")
    # scoreAvg = np.nanmean(scoreAll, axis=1)
    # for metric, score in zip(metrics, scoreAvg):
    #     print(metric, score)
    # print("score")
    # print(material_index)
    # print(len(material_index))
    scoreAvg = np.mean(scoreAll, axis=1)
    result={}
    for metric, score in zip(metrics, scoreAvg):
        result[metric]=score
        print(metric, score)
    return result


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

if __name__ == '__main__':
    # METRICS_MAP = ['MAP',  'NDCG', 'HIT']
    # gt_doc_ids = {0, 1, 2}
    # pred_doc_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # result = cal_metrics(
    #     gt=gt_doc_ids, pred=pred_doc_ids, metrics_map=METRICS_MAP)
    # print(result)
    # gt_doc_ids = {0, 1, 2}
    # pred_doc_ids = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # result = cal_metrics(
    #     gt=gt_doc_ids, pred=pred_doc_ids, metrics_map=METRICS_MAP,maxnum=5)
    # print(result)
    # getScore('resource/xunfei/PreSumm.result','resource/xunfei/PreSumm.gold')
    # getScore("resource/geo_q_yzhao/bm25/answer.txt", "resource/geo_q_yzhao/bm25/test.tgt")
    # getScore("resource/geo_all/logs/PreSumm.result", "resource/raw/test.tgt")
    # getScore("resource/geo_q/logs/PreSumm.result", "resource/raw/test.tgt")
    # getScore("resource/geo_q_m/logs/PreSumm.result", "resource/raw/test.tgt")
    # getScore("resource/geo_q_m/logs_new/PreSumm.result", "resource/raw/test.tgt")
    # getScore("resource/geo_q_m/logs_baseline/PreSumm.result", "resource/raw/test.tgt")
    # getScore("resource/roberta_q_m/logs/PreSumm.result", "resource/raw/test.tgt")
    # getScore("resource/raw/material", "resource/raw/test.tgt")
    # getScore("resource/geo_q_m_image/q_logs/PreSumm.result", "resource/geo_q_m_image/q_logs/gold.result")
    # getScore("resource/geo_q_m_image/q_i_copy/PreSumm.result", "resource/geo_q_m_image/q_logs/gold.result")

    # getScore("resource/geo_q_m_image/q_i_logs/PreSumm.result", "resource/geo_q_m_image/q_logs/gold.result")
    # getScore('graph_data/results/.4000.candidate','graph_data/results/.4000.gold')
    getScore('wikihow_data/results_splitgen/.1.candidate','wikihow_data/results_splitgen/.1.gold',language='en')