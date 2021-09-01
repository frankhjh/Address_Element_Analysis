#!/usr/bin/env python
import json
from data_reader import read_conll_file,read_txt_file

def gen_addrs(train_file,dev_file,test_file):
    all_addrs=[]
    
    train_set=[(words,labels) for (words,labels) in read_conll_file(train_file)]
    dev_set=[(words,labels) for (words,labels) in read_conll_file(dev_file)]
    test_set=read_txt_file(test_file)

    for item in train_set+dev_set:
        all_addrs.append(item[0])
    
    for addr in test_set:
        all_addrs.append(addr)
    with open('../output/addr.json','w',encoding='utf-8') as f:
        json.dump(all_addrs,f,ensure_ascii=False)

