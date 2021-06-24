from utils.data_reader import read_conll_file,read_txt_file
from utils.data_loader import Corpus_Dataset
import torch

#prepare data for attention model
def prepare_data_attn(train_file,dev_file,test_file):
    
    #train set
    train_set=[(words,tag_classes) for (words,tag_classes) in read_conll_file(train_file)]
    #dev set
    dev_set=[(words,tag_classes) for (words,tag_classes) in read_conll_file(dev_file)]
    #test set
    test_set=read_txt_file(test_file)

    ###################### deal with token first! ############################
    train_tokens_li=[train_set[i][0] for i in range(len(train_set))]
    dev_tokens_li=[dev_set[i][0] for i in range(len(dev_set))]
    tokens_li=train_tokens_li+dev_tokens_li+test_set

    token_set=set()
    for i in tokens_li:
        token_set.update(i)

    unique_token_li=sorted(list(token_set))
    
    #build dict with padding
    token2idx={'P':len(unique_token_li)}
    for i,token in enumerate(unique_token_li):
        token2idx[token]=i
    
    #figure out the max length
    l=len(unique_token_li)
    max_train=max([len(train_x[i]) for i in range(len(train_x))])
    max_dev=max([len(dev_x[i]) for i in range(len(dev_x))])
    max_test=max([len(test_x[i]) for i in range(len(test_x))])
    MAX_LEN=max(max_train,max_dev,max_test)

    #do the padding
    train_x_pad=[[token2idx_2[token] for token in seq]+[l]*(MAX_LEN-len(seq)) for seq in train_tokens_li]
    dev_x_pad=[[token2idx_2[token] for token in seq]+[l]*(MAX_LEN-len(seq)) for seq in dev_tokens_li]
    test_x_pad=[[token2idx_2[token] for token in seq]+[l]*(MAX_LEN-len(seq)) for seq in test_set]

    ############################ do the same for tag! ################################
    train_tag_classes_li=[train_set[i][1] for i in range(len(train_set))]
    dev_tag_classes_li=[dev_set[i][1] for i in range(len(dev_set))]

    tag_classes_li=train_tag_classes_li+dev_tag_classes_li
    tag_classes_set=set()

    for i in tag_classes_li:
        tag_classes_set.update(i)

    unique_tag_classes_list=sorted(list(tag_classes_set))
    
    #build dict with padding
    tc2idx={'P':len(unique_tag_classes_list)} 
    for i,item in enumerate(unique_tag_classes_list):
        tc2idx[item]=i

    l2=len(unique_tag_classes_list)
    train_y=[[tc2idx[tag_class] for tag_class in tag_classes]+[l2]*(MAX_LEN-len(tag_classes)) for tag_classes in train_tag_classes_li]
    dev_y=[[tc2idx[tag_class] for tag_class in tag_classes]+[l2]*(MAX_LEN-len(tag_classes)) for tag_classes in dev_tag_classes_li]
                







