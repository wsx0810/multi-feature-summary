# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:03:26 2018

@author: wu
"""

import utils
import codecs
import rouge
import re

if __name__ == '__main__':
#    utils.preprocess()
    sen_list,sen_list_new = utils.extract_sentence('data/all_article_seg.txt',extract_sen_num=3)
#    sen_list_new = utils.extract_sentence_new('data/all_article_seg.txt',extract_sen_num=3)
#    sen_list1 = utils.extract_sen_compress('data/all_article_seg.txt',sen_num=3)
    file = codecs.open('data/all_summary.txt','r','utf-8')
    refs = file.readlines()
    sen_list_com = []
    sen_list_new_com = []
    for sen in sen_list:
        sen = re.sub(r'((\d{4}[-/])?\d{1,2}[-/]\d{1,2}(\s\d{1,2}:\d{1,2})?)|\d{1,2}:\d{1,2}:\d{1,2}','',sen)
        sen = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|<.*?>",'',sen)
        sen = re.sub(r'(\d{4}年)?\d{1,2}月\d{1,2}日(\d{1,2}时\d{1,2}分)?','',sen)
        sen_list_com.append(sen)
    for sen in sen_list_new:
        sen = re.sub(r'((\d{4}[-/])?\d{1,2}[-/]\d{1,2}(\s\d{1,2}:\d{1,2})?)|\d{1,2}:\d{1,2}:\d{1,2}','',sen)
        sen = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|<.*?>",'',sen)
        sen = re.sub(r'(\d{4}年)?\d{1,2}月\d{1,2}日(\d{1,2}时\d{1,2}分)?','',sen)
        sen_list_new_com.append(sen)

#    print('no_compress:\n',rouge.rouge(sen_list,refs))
    print('no_compress:\n',rouge.rouge(sen_list,refs))
    print('no_compress_new:\n',rouge.rouge(sen_list_new,refs))
    print('no_compress_com:\n',rouge.rouge(sen_list_com,refs))
    print('no_compress_new_com:\n',rouge.rouge(sen_list_new_com,refs))
#    print('compress:\n',rouge.rouge(sen_list1,refs))
    utils.nlp.close()
    file.close()
#    print(sen_list)
#    print(sen_list1)    