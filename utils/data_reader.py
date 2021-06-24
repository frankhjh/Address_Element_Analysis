#train/dev data loader

def read_conll_file(file_name):
        current_words = []
        current_tag_classes = []
        with open(file_name, encoding='utf-8') as conll:
            for line in conll:
                line = line.strip()
                if line:
                    word,tag_class = line.split(' ')
                    current_words.append(word)
                    current_tag_classes.append(tag_class)

                else:
                    yield (current_words, current_tag_classes)
                    current_words = []
                    current_tag_classes = []
            #in case last line of file is not empty
            if current_tag_classes != []:
                yield (current_words, current_tag_classes) 

#test data loader

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