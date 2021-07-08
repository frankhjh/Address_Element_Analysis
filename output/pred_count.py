#!/usr/bin/env python

def done_count(out_path):
    i=0
    with open(out_path,'r',encoding='utf-8') as f:
        for i,line in enumerate(f):
            i+=1
    return i
