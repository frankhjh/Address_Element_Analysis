#!/usr/bin/env python
from data_reader import read_conll_file,read_txt_file
from generate_addrs import gen_addrs
from generate_labels import gen_labels
import torch
import json
import os

# dict store all tokens of address words
def build_word_dict(src_files):
    if not os.path.exists('../output/addr.json'):
        gen_addrs(src_files[0],src_files[1],src_files[2])
    
    with open('../output/addr.json','r',encoding='utf-8') as f:
        addr_li=json.load(f)
    
    addr_tokens=set()
    for addr in addr_li:
        addr_tokens.update(addr)
    
    dic={'PAD':0}
    for i in addr_tokens:
        dic[i]=len(dic) 
    return dic
# dict store all tokens of address labels
def build_label_dict(src_files):
    if not os.path.exists('../output/labels.json'):
        gen_labels(src_files[0],src_files[1])
    
    with open('../output/labels.json','r',encoding='utf-8') as f:
        label_li=json.load(f)
    
    dic={'PAD':0,'<START>':1,'<STOP>':2}
    for i in label_li:
        dic[i]=len(dic) 
    return dic

def token2id(tokens,dic):
    ids=[]
    for token in tokens:
        ids.append(dic.get(token))
    return ids

# convert both words and labels into ids representation
def convert_train_dev_sample(sample):
    words,labels=sample
    word_ids=token2id(words,words_dict)
    label_ids=token2id(labels,labels_dict)
    
    return word_ids,label_ids

def convert_test_sample(sample):
    words=sample
    word_ids=token2id(words,words_dict)
    return word_ids

    
if __name__=='__main__':
    words_dict=build_word_dict(('../data/train.conll','../data/dev.conll','../data/final_test.txt'))
    labels_dict=build_label_dict(('../data/train.conll','../data/dev.conll'))

    train_set=[(words,labels) for (words,labels) in read_conll_file('../data/train.conll')]
    dev_set=[(words,labels) for (words,labels) in read_conll_file('../data/dev.conll')]
    test_set=read_txt_file('../data/final_test.txt')

    encoded_train_set=list(map(convert_train_dev_sample,train_set))
    encoded_dev_set=list(map(convert_train_dev_sample,dev_set))
    encoded_test_set=list(map(convert_test_sample,test_set))

    print(encoded_train_set[0],'\n')
    print(encoded_dev_set[0],'\n')
    print(encoded_test_set[0])



    
