#!/usr/bin/env python
from utils.data_reader import read_txt_file

def pred_slice(tmp_pred,final_pred,test_file):
    test_set=read_txt_file(test_file)
    
    tmp_pred=[]
    final_pred=[]
    with open(tmp_pred,'r',encoding='utf-8') as tmp_f:
        for line in tmp_f:
            line=line.strip()
            tmp_pred.append(line.split(' '))
    tmp_f.close()
    #slice
    for i in range(len(test_set)):
        tmp_pred[i]=tmp_pred[i][:len(test_set[i])]
        final_pred.append(' '.join(tmp_pred[i]))
    
    with open(final_pred,'w',encoding='utf-8') as final_f:
        for p in final_pred:
            final_f.write(p+'\n')
    final_f.close()


def submit(src_file,tmp_pred_file,final_pred_file,tar_file):
    #first call the pred_slice()
    pred_slice(tmp_pred_file,final_pred_file,src_file)

    output=[]
    with open(src_file,'r',encoding='utf-8') as f1,open(final_pred_file,'r',encoding='utf-8') as f2:
        for src,pred in zip(f1,f2):
            src,pred=src.strip(),pred.strip()
            line=src+'\u0001'+pred
            output.append(line)
    f1.close()
    f2.close()

    with open(tar_file,'w',encoding='utf-8') as f:
        for item in output:
            f.write(item+'\n')
    f.close()

if __name__=='__main__':
    submit('data/final_test.txt','output/tmp_pred.txt','output/final_pred.txt','output/胡小白_addr_parsing_runid.txt)
