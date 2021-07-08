from utils.data_reader import read_conll_file,read_txt_file
import torch

def build_set(train_file,dev_file,test_file,src=True): #whether for src or tar
    #load data
    train_set=[(words,tag_classes) for (words,tag_classes) in read_conll_file(train_file)]
    dev_set=[(words,tag_classes) for (words,tag_classes) in read_conll_file(dev_file)]
    test_set=read_txt_file(test_file)
    
    #figure out the max len of sequences among train/dev/test set
    max_train=max([len(train_set[i]) for i in range(len(train_set))])
    max_dev=max([len(dev_set[i]) for i in range(len(dev_set))])
    max_test=max([len(test_set[i]) for i in range(len(test_set))])
    MAX_LEN=max(max_train,max_dev,max_test)

    #build set
    s=set()
    if src:
        src_train_li=[train_set[i][0] for i in range(len(train_set))]
        src_dev_li=[dev_set[i][0] for i in range(len(dev_set))]
        src_li=src_train_li+src_dev_li+test_set
        for src in src_li:
            s.update(src)
        return MAX_LEN,sorted(list(s)),src_train_li,src_dev_li,test_set
    else:
        tar_train_li=[train_set[i][1] for i in range(len(train_set))]
        tar_dev_li=[dev_set[i][1] for i in range(len(dev_set))]
        tar_li=tar_train_li+tar_dev_li

        for tar in tar_li:
            s.update(tar)
        return MAX_LEN,sorted(list(s)),tar_train_li,tar_dev_li
    
def mapping(li,src=True):
    if src:
        src2idx={'P':len(li)}
        for i,token in enumerate(li):
            src2idx[token]=i
        return src2idx
    tar2idx={'P':len(li),'SOS':len(li)+1,'EOS':len(li)+2}
    for i,tag in enumerate(li):
        tar2idx[tag]=i
    return tar2idx

def prepare_data(train_file,dev_file,test_file,src=True):
    
    if src:
        max_len,src_set,src_train_li,src_dev_li,test_set=build_set(train_file,dev_file,test_file,src=True)
        l=len(src_set)
        m=mapping(src_set,src=True)
        
        #do the padding
        train_x_pad=[[m[token] for token in seq]+[l]*(max_len-len(seq)) for seq in src_train_li]
        dev_x_pad=[[m[token] for token in seq]+[l]*(max_len-len(seq)) for seq in src_dev_li]
        test_x_pad=[[m[token] for token in seq]+[l]*(max_len-len(seq)) for seq in test_set]

        return (torch.LongTensor(train_x_pad),torch.LongTensor(dev_x_pad),torch.LongTensor(test_x_pad)),m,src_set,test_set
    
    max_len,tar_set,tar_train_li,tar_dev_li=build_set(train_file,dev_file,test_file,src=False)
    l=len(tar_set)
    m=mapping(tar_set,src=False)
    
    train_y_pad=[[l+1]+[m[tag_class] for tag_class in seq]+[l]*(max_len-len(seq))+[l+2] for seq in tar_train_li]
    dev_y_pad=[[l+1]+[m[tag_class] for tag_class in seq]+[l]*(max_len-len(seq))+[l+2] for seq in tar_dev_li]

    return (torch.LongTensor(train_y_pad),torch.LongTensor(dev_y_pad)),m,tar_set,max_len
                








