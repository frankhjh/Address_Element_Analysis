#!/usr/bin/env python
import json
from data_reader import read_conll_file,read_txt_file

def gen_labels(train_file,dev_file):
    all_labels=set()
    
    train_set=[(words,labels) for (words,labels) in read_conll_file(train_file)]
    dev_set=[(words,labels) for (words,labels) in read_conll_file(dev_file)]

    for item in train_set+dev_set:
        all_labels.update(item[1])
    
    with open('../output/labels.json','w') as f:
        json.dump(list(all_labels),f)

