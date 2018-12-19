# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:23:32 2018

@author: wu
"""
import codecs
import json
import stanfordcorenlp
import nltk
import re
import rouge
import sys
nlp = stanfordcorenlp.StanfordCoreNLP(r'E:\wsx file\Intresting\experiment\download_cord\stanford-corenlp-full-2018-02-27',lang='zh')
tparse = nltk.tree.Tree
    
def json2obj(filepath):
    # ½«Ò»¸öÎÄ¼þÖÐµÄjsonÈ«²¿×ª»»Îªpython dict£¬´æÈëÁÐ±í
    infile = codecs.open(filepath, 'r', 'utf-8')
    jss = []
    for line in infile.readlines():
        try:
            jss.append(json.loads(line.strip()))
        except  :
            print(line)

    infile.close()
    return jss

def data_split_seg(filepath):
    jsons = json2obj(filepath)
    all_article_seg = codecs.open('data/all_article_seg.txt','w','utf-8')
    all_summary_seg = codecs.open('data/all_summary_seg.txt','w','utf-8')
    all_article = codecs.open('data/all_article.txt','w','utf-8')
    all_summary = codecs.open('data/all_summary.txt','w','utf-8')
    for jsonf in jsons:
        art = jsonf['article']
        art = art.replace(u'<Paragraph>','')
        art = art.replace(u'>>','')
        art = art.replace(u'?','？')
        art = art.replace(u'？？','？')
        art = art.replace(u'!','！')
        art = art.replace(u'(','（')
        art = art.replace(u')','）')
        art = art.replace(u'！！！','！')
        art = art.replace(u'。。','。')
        art = art.replace(u'！？','')
        art = art.replace(u'？！','')
        art = art.replace(u'。”。','。”')
        all_article.write(art + '\n')
        art_seg = nlp.word_tokenize(art)
        summ = jsonf['summarization']
        summ = summ.replace(u'(图)', '')
        summ = summ.replace(u'（图）', '')
        summ = summ.replace(u'(组图)', '')
        summ = summ.replace(u'（组图）', '')
        summ = summ.replace(u'\n', '')
        summ = summ.replace(u'\r\n', '')
        summ = summ.replace('\n', '')
        summ = summ.replace('\r\n', '')
        all_summary.write(summ + '\n')        
        summ_seg = nlp.word_tokenize(summ)
        for i in range(len(art_seg)):
            if i == len(art_seg) - 1:
                all_article_seg.write(art_seg[i] + '\n')
            else:
                all_article_seg.write(art_seg[i] + ' ')
        for j in range(len(summ_seg)):
            if j == len(summ_seg) - 1:
                all_summary_seg.write(summ_seg[j] + '\n')
            else:
                all_summary_seg.write(summ_seg[j] + ' ')
    all_article.close()
    all_summary.close()
    all_article_seg.close()
    all_summary_seg.close()
#    nlp.close()
    return 'data/all_article_seg.txt','data/all_summary_seg.txt','data/all_article.txt','data/all_summary.txt'

#返回值是[[sen1,sen2...],[sen1,sen2...]...]格式
def sentence_split(filepath,accord='[。？！]'):
    file = codecs.open(filepath,'r','utf-8')
    sentence_list = []
    for article in file.readlines(): 
        article = article.strip()
        sentence_list.append(re.split(accord,article))
    file.close()
    return sentence_list 
#以上返回所有按句号、问号和感叹号划分的句子列表，格式为[[sen1,sen2...],[]...]
    
def normalization (ori_list):
    normalizied_list = []
    for i in ori_list:
        total = sum(i)
        temp_list = []
        for j in i:
            temp_list.append(j/total)
        normalizied_list.append(temp_list)
    return normalizied_list

#def sen2parse(sentence):
#    from stanfordcorenlp import StanfordCoreNLP
#    nlp = StanfordCoreNLP(r'E:\wsx file\Intresting\experiment\download_cord\stanford-corenlp-full-2018-02-27',lang='zh',memory='4g')
#    parse_str = nlp.parse(sentence)    
##    nlp.close() # Do not forget to close! The backend server will consume a lot memery.
#    return parse_str

def dps_sen(trees,filterlabels,keywordpos):
    temp_str = ''
    for tree in trees:
        if tree.label() not in filterlabels:
            if tree.height() == 2:
                temp_str += tree[0]
#                print(tree[0],end='') #输出剪枝后的句子                
            else:
                temp_str += dps_sen(tree,filterlabels,keywordpos)

    return temp_str

def sentence_compress(sentence):
    filterlabels = ['OD','CLP','LCP','LC','DEG','PP','P','AS']
    keywordpos = ['NN','VV','AD','ADJ','JJ','NR']
    parse_str = nlp.parse(sentence)
    parse_str = parse_str.replace('\r\n',' ')
    tree = tparse.fromstring(parse_str)
#    tree.draw()
    compress_sen = dps_sen(tree,filterlabels,keywordpos)
#    nlp.close()
    return compress_sen   
    
def dps(trees,filterlabels,keywordpos):
    keywords = []
    for tree in trees:
        if tree.label() not in filterlabels:
            if tree.height() == 2:
#                print(tree[0],end='') #输出剪枝后的句子
                if tree.label() in keywordpos:
                    keywords.append(tree[0])
            else:
                keywords += dps(tree,filterlabels,keywordpos)

    return keywords
    
#返回值为sen-n中的关键词
def extract_keywords(parse_str):
#    parse_str = parse_str.replace('\r\n',' ')
    tree = tparse.fromstring(parse_str)
#    tree.draw()
    filterlabels = ['OD','CLP','LCP','LC','DEG','PP','P','AS']
    keywordpos = ['NN','VV','AD','ADJ','JJ','NR']
#    tree.draw()
    keywords = dps(tree,filterlabels,keywordpos)
#    print(dps(tree,filterlabels,keywordpos))
    return keywords
#    
def TF_score(filepath):
    all_art_keywords = []    
    sentence_list = sentence_split(filepath,accord='[。？！]')
#    print(sentence_list)
    for art in sentence_list:
        art_keywords = []
        for sen in art:
            sen = sen.strip()
            if sen != '':
#                print(sen)
                try:
                    parse_str = nlp.parse(sen)
                except:
                    print(art)
                    print('aaaaa',sen,'aaaa',len(sen))
                    sys.exit()
                art_keywords += extract_keywords(parse_str)
        art_keywords = list(set(art_keywords))
        all_art_keywords.append(art_keywords)
    #以上返回所有关键词，格式为[[w1,w2...],[w1,w2...]...]
    file = codecs.open(filepath,'r','utf-8')
    doc = file.readlines()
    all_sen_score = []
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    for i in range(len(sentence_list)):
        leng = len(doc[i])
        sen_score = []      
        for sen in sentence_list[i]:
            if sen != '':
                score = 0
                for w in all_art_keywords[i]:
                    fre = doc[i].count(w) * 100 /leng           
                    score += sen.count(w) * fre / len(sen)
                sen_score.append(score)
        all_sen_score.append(sen_score) 
    #以上返回所有句子的TF值，格式为[[sen1_TF,sen2_tf...],[]...]     
#    nlp.close()
    file.close()
    return normalization(all_sen_score)

from gensim.models import word2vec

import numpy as np
def w2v(filepath,num_feature):
    w2v_sentences = word2vec.Text8Corpus(filepath)
    model=word2vec.Word2Vec(w2v_sentences,min_count=1,size=num_feature)
#    print(model.wv.index2word)
    model.save('data/model')

#model = word2vec.Word2Vec.load('data/model')
#index2word_set = set(model.wv.index2word)

def avg_feature_vector(sentence,model, index2word_set,num_features=200):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ))
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def vec_cosine(vector1,vector2):
    score = (1 + np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))) / 2
    return score
import os
def similarity_score(filepath,num_feature=200):
    if os.path.exists('data/modle'):
        pass
    else:
        w2v(filepath,num_feature)
    model = word2vec.Word2Vec.load('data/model')
    index2word_set = set(model.wv.index2word)
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    all_vec_list = []
    for art in sentence_list:
        vec_list = []
        for sen in art:
            if sen != '':                
                feature_vec = avg_feature_vector(sen,model,index2word_set,num_feature)
                vec_list.append(feature_vec)
        all_vec_list.append(vec_list)
    #以上返回所有句子的向量，格式为[[sen1_vec,sen2_vec...],[]...]
    all_sim_score_list = []
    for art_vec in all_vec_list:
        sim_score_list = []
        for sen_vec in art_vec:
            sim_score = 0
            for sen_vec_cp in art_vec:
                sim = vec_cosine(sen_vec,sen_vec_cp)
#                print(sim)
                sim_score += sim
            sim_score_list.append(sim_score-1)
        all_sim_score_list.append(sim_score_list)
    #以上返回所有句子的向量相似度得分，格式为[[sen1_vec_simscore,sen2_vec_simscore...],[]...]
    return normalization(all_sim_score_list)

import math
pi = math.pi
def length_score(filepath):
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    all_len_score = []
    for art in sentence_list:
        count = 0
        mid_score = 0
        for sen in art:
            if sen != '':
                sen = sen.replace(' ','')
                count +=1
                mid_score += (len(sen)-45)**2
        sita = math.sqrt(mid_score/count)
        
        len_score = []
        for sen in art:
            if sen != '':
                sen = sen.replace(' ','')
                score = math.exp(-(len(sen)-30)**2/(2*sita**2))/(math.sqrt(2*pi)*sita)
                len_score.append(score)
        all_len_score.append(len_score)
        
    return normalization(all_len_score)

def location_score(filepath):
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    all_loc_score = []
    for art in sentence_list:
        count = 0
        loc_score = []
        for sen in art:
            if sen != '':
                count +=1
        sen_num = count
        loc = 0
        for sen in art:
            if sen != '':
                loc += 1
                score = (sen_num-loc+1)/sen_num
                loc_score.append(score)
        all_loc_score.append(loc_score)
    return normalization(all_loc_score)

loc_dic = {1.0: 0.089999999999999997, 2.0: 0.10400000000000001, 3.0: 0.096000000000000002, 4.0: 0.069000000000000006, 5.0: 0.050000000000000003, 6.0: 0.043000000000000003, 7.0: 0.027, 8.0: 0.029000000000000001, 9.0: 0.036000000000000004, 10.0: 0.021999999999999999, 11.0: 0.016, 12.0: 0.021000000000000001, 13.0: 0.019, 14.0: 0.017000000000000001, 15.0: 0.018000000000000002, 16.0: 0.019, 17.0: 0.02, 18.0: 0.0090000000000000011, 19.0: 0.016, 20.0: 0.0080000000000000002, 21.0: 0.0080000000000000002, 22.0: 0.0080000000000000002, 23.0: 0.0050000000000000001, 24.0: 0.0030000000000000001, 25.0: 0.010999999999999999, 26.0: 0.0090000000000000011, 27.0: 0.0040000000000000001, 28.0: 0.0080000000000000002, 29.0: 0.0050000000000000001, 30.0: 0.01, 31.0: 0.0060000000000000001, 32.0: 0.0070000000000000001, 33.0: 0.0060000000000000001, 34.0: 0.0070000000000000001, 35.0: 0.0080000000000000002, 36.0: 0.0050000000000000001, 37.0: 0.0050000000000000001, 38.0: 0.0030000000000000001, 39.0: 0.0060000000000000001, 40.0: 0.001, 41.0: 0.0040000000000000001, 42.0: 0.0050000000000000001, 43.0: 0.0060000000000000001, 44.0: 0.0050000000000000001, 45.0: 0.0070000000000000001, 46.0: 0.001, 47.0: 0.0030000000000000001, 48.0: 0.001, 49.0: 0.001, 50.0: 0.0060000000000000001}

def location_score_new(filepath):
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    all_loc_score = []
    for art in sentence_list:
        loc_score = []
        loc = 0
        for sen in art:
            if sen != '':
                loc += 1
                loc_score.append(((loc-len(art))**2+0.00001)/((1-len(art))**2+0.00001))
#                if loc in loc_dic.keys():
#                    loc_score.append(loc_dic[loc])
#                else:
#                    loc_score.append(0.001)
                
#                if loc == 1:
#                    score = 0.27
#                else:
#                    score = 0.213/loc+0.243/(loc**2)
#                loc_score.append(score)
        all_loc_score.append(loc_score)
    return normalization(all_loc_score)

def avg_rouge_score(preds, trues):
    rouge_1 = rouge.rouge_n(preds, trues,1)
    rouge_2 = rouge.rouge_n(preds, trues,2)
    rouge_l = rouge.rouge_l_sentence_level(preds, trues)
    rouge_avg = zip(rouge_1,rouge_2,rouge_l)
    avg_score,avg_p,avg_r = map(np.mean,rouge_avg)
#    avg_score = (result['rouge_1/f_score'] + result['rouge_2/f_score'] + result['rouge_L/f_score']) / 3
#    print('rouge_1:' + str(rouge_1) + 'rouge_2:' + str(rouge_2) + 'rouge_L:' + str(rouge_l))
    return avg_score

def extract_sentence(filepath,parm=[1,0.5,0.5,2],parm_new=[2,9,2,4],extract_sen_num=1,feature_dim=200):
#    file = codecs.open('data/statistics.txt','w','utf-8')
#    file_summ = codecs.open('data/all_summary.txt','r','utf-8')
#    summary = file_summ.readlines()
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    all_sen_score = TF_score(filepath)
    all_sim_score = similarity_score(filepath,num_feature=feature_dim)
    all_len_score = length_score(filepath)
    all_loc_score = location_score(filepath)
    all_loc_score_new = location_score_new(filepath)
    all_sentence_score = []
    all_sentence_score_new = []
    for i in range(len(sentence_list)):
        sen_score_dic = {}
        sen_score_dic_new = {}
        for j in range(len(sentence_list[i])):
            score = 0
            score_new = 0
            if sentence_list[i][j] != '':
#                sentence_pre = sentence_list[i][j].replace(' ','')
#                rouge_score = avg_rouge_score(sentence_pre,summary[i])
#                file.write(str(i)+' '+str(j)+' '+str(all_sen_score[i][j])+' '+str(all_sim_score[i][j])+' '+str(all_len_score[i][j])+' '+str(all_loc_score[i][j])+' '+str(rouge_score)+'\n')
                score += parm[0]*all_sen_score[i][j]+parm[1]*all_sim_score[i][j]+parm[2]*all_len_score[i][j]+parm[3]*all_loc_score[i][j]
                score_new += parm_new[0]*all_sen_score[i][j]+parm_new[1]*all_sim_score[i][j]+parm_new[2]*all_len_score[i][j]+parm_new[3]*all_loc_score_new[i][j]
                sen_score_dic[sentence_list[i][j]] = score
                sen_score_dic_new[sentence_list[i][j]] = score_new
        all_sentence_score.append(sen_score_dic)
        all_sentence_score_new.append(sen_score_dic_new)
#    file.close()
#    file_summ.close()
    #以上返回句子最后得分，格式为[{sen1:score,sen2:score...},{}...]
    all_extract_sentence = []
    for art in all_sentence_score:
        temp_str = ''
        sorted_list = sorted(art.items(),key = lambda x:x[1],reverse=True)
        for i in range(len(sorted_list[:3])):
            temp_str += sorted_list[i][0]+'，'
        temp_str=temp_str[:-1] + '\n'
#        temp_str.replace(' ','')
        all_extract_sentence.append(temp_str.replace(' ',''))
        
    all_extract_sentence_new = []
    for art in all_sentence_score_new:
        temp_str = ''
        sorted_list = sorted(art.items(),key = lambda x:x[1],reverse=True)
        for i in range(len(sorted_list[:3])):
            temp_str += sorted_list[i][0]+'，'
        temp_str=temp_str[:-1] + '\n'
#        temp_str.replace(' ','')
        all_extract_sentence_new.append(temp_str.replace(' ',''))
    #以上返回抽取的句子，格式为['sen1','sen2'...] ，列表长度为文本数目 
    return all_extract_sentence,all_extract_sentence_new

def extract_sentence_new(filepath,parm=[2,9,2,4],extract_sen_num=1,feature_dim=200):
#    file = codecs.open('data/statistics.txt','w','utf-8')
#    file_summ = codecs.open('data/all_summary.txt','r','utf-8')
#    summary = file_summ.readlines()
    sentence_list = sentence_split(filepath,accord='[，。？！]')
    all_sen_score = TF_score(filepath)
    all_sim_score = similarity_score(filepath,num_feature=feature_dim)
    all_len_score = length_score(filepath)
#    all_loc_score = location_score(filepath)
    all_loc_score = location_score_new(filepath)
    all_sentence_score = []
    for i in range(len(sentence_list)):
        sen_score_dic = {}
        for j in range(len(sentence_list[i])):
            score = 0
            if sentence_list[i][j] != '':
#                sentence_pre = sentence_list[i][j].replace(' ','')
#                rouge_score = avg_rouge_score(sentence_pre,summary[i])
#                file.write(str(i)+' '+str(j)+' '+str(all_sen_score[i][j])+' '+str(all_sim_score[i][j])+' '+str(all_len_score[i][j])+' '+str(all_loc_score[i][j])+' '+str(rouge_score)+'\n')
                score += parm[0]*all_sen_score[i][j]+parm[1]*all_sim_score[i][j]+parm[2]*all_len_score[i][j]+parm[3]*all_loc_score[i][j]
                sen_score_dic[sentence_list[i][j]] = score
        all_sentence_score.append(sen_score_dic)
#    file.close()
#    file_summ.close()
    #以上返回句子最后得分，格式为[{sen1:score,sen2:score...},{}...]
    all_extract_sentence = []
    for art in all_sentence_score:
        temp_str = ''
        sorted_list = sorted(art.items(),key = lambda x:x[1],reverse=True)
        for i in range(len(sorted_list[:3])):
            temp_str += sorted_list[i][0]+'，'
        temp_str=temp_str[:-1] + '\n'
#        temp_str.replace(' ','')
        all_extract_sentence.append(temp_str.replace(' ',''))
    #以上返回抽取的句子，格式为['sen1','sen2'...] ，列表长度为文本数目 
    return all_extract_sentence

def extract_sen_compress(filepath,parm=[1,0.5,0.5,0.1],sen_num=1,feature_dim=200):
    all_extract_sentence = extract_sentence(filepath,parm=parm,extract_sen_num=sen_num,feature_dim=feature_dim)
    final_compressed_sentence = []
    for sen in all_extract_sentence:
        sen_compressed = sentence_compress(sen)
        final_compressed_sentence.append(sen_compressed)
    return final_compressed_sentence

def count_words_num(filepath):
    with open(filepath,encoding='utf-8') as file:
        content = file.read()
        total = len(content)
        lines = len(re.split('[，。？！\n]',content))
        avg_words = total/lines
    return avg_words

def cal_avg_sen_words(filepath):
    a_s,s_s,ar,su = data_split_seg(filepath)
    result = count_words_num(ar)
    with open(su,encoding = 'utf-8') as file:
        content = file.read()
        result1 = len(content)/len(content.split('\n'))
    return result,result1
#以上分别返回文章和摘要平均每句的字数
def preprocess():
    data_split_seg('data/train_with_summ_2000.txt')
    
if __name__ == "__main__":
    pass
    
    
#    print(TF_score('data/all_article_seg.txt'))
#    nlp.close()
#    w2v()
#    print(similarity_score('data/10.txt'))
#    print(length_score('data/10.txt'))
#    print(location_score('data/10.txt'))
#    print(extract_sen_compress('data/10.txt'))
#    print(sentence_compress('我们的本领有适应的一面，也有不适应的一面'))
