#!/usr/bin/env python

# train/dev data reader
def read_conll_file(file_name):
        current_words = []
        current_labels = []
        with open(file_name, encoding='utf-8') as conll:
            for line in conll:
                line = line.strip()
                if line:
                    word,label = line.split(' ')
                    current_words.append(word)
                    current_labels.append(label)

                else:
                    yield (current_words, current_labels)
                    current_words = []
                    current_labels = []
            #in case last line of file is not empty
            if current_labels != []:
                yield (current_words, current_labels) 

# test data reader
def read_txt_file(file_name):
    addrs=[]
    with open(file_name,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            idx,addr=line.split('\u0001')
            addrs.append(addr)
    n=len(addrs)
    for i in range(n):
        addrs[i]=[addrs[i][j] for j in range(len(addrs[i]))]
        
    return addrs  

if __name__=='__main__':
    train_set=[(words,labels) for (words,labels) in read_conll_file('../data/train.conll')]
    dev_set=[(words,labels) for (words,labels) in read_conll_file('../data/dev.conll')]
    test_set=read_txt_file('../data/final_test.txt')

    print(train_set[0],'\n')
    print(dev_set[0],'\n')
    print(test_set[0])