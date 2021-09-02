import torch
from torch.utils.data import DataLoader
from utils.data_reader import read_conll_file,read_txt_file
from utils.dataset import Train_Dev_Dataset,Test_Dataset
from utils.encoding import build_word_dict,build_label_dict,token2id,convert_train_dev_sample,convert_test_sample

def get_max_len(src_files):
    words_dict,vocab_size=build_word_dict((src_files[0],src_files[1],src_files[2]))
    labels_dict,label_size=build_label_dict((src_files[0],src_files[1]))

    train_set=[(words,labels) for (words,labels) in read_conll_file(src_files[0])]
    dev_set=[(words,labels) for (words,labels) in read_conll_file(src_files[1])]
    test_set=read_txt_file(src_files[2])

    encoded_train_set,encoded_dev_set,encoded_test_set=[],[],[]
    for i in range(len(train_set)):
        encoded_sample=convert_train_dev_sample(train_set[i],words_dict,labels_dict)
        encoded_train_set.append(encoded_sample)
    
    for i in range(len(dev_set)):
        encoded_sample=convert_train_dev_sample(dev_set[i],words_dict,labels_dict)
        encoded_dev_set.append(encoded_sample)
    
    for i in range(len(test_set)):
        encoded_sample=convert_test_sample(test_set[i],words_dict)
        encoded_test_set.append(encoded_sample)

    max_train_len=max([len(i[0]) for i in encoded_train_set])
    max_dev_len=max([len(i[0]) for i in encoded_dev_set])
    max_test_len=max([len(i) for i in encoded_test_set])

    return max(max_train_len,max_dev_len,max_test_len),words_dict,vocab_size,labels_dict,label_size,encoded_train_set,encoded_dev_set,encoded_test_set

def prep_data(type,max_len,data):
    if type in ('train','dev'):
        x=[data[i][0]+[0]*(max_len-len(data[i][0])) for i in range(len(data))]
        x=torch.tensor(x,dtype=torch.long)
        y=[data[i][1]+[0]*(max_len-len(data[i][1])) for i in range(len(data))]
        y=torch.tensor(y,dtype=torch.long)
        
        if type=='train':
            trainset=Train_Dev_Dataset(x,y)
            return DataLoader(trainset,batch_size=32,shuffle=True)
        else:
            devset=Train_Dev_Dataset(x,y)
            return DataLoader(devset,batch_size=32,shuffle=False)
    # else must be 'test'
    x=[data[i]+[0]*(max_len-len(data[i])) for i in range(len(data))]
    x=torch.tensor(x,dtype=torch.long)
    testset=Test_Dataset(x)
    return DataLoader(testset,shuffle=False)






