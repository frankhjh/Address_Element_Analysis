from model.lstm_crf import BiLSTM_CRF
from prepare_data import get_max_len,prep_data
import torch
import torch.optim as optim
from tqdm import tqdm
import json

def validate(model,dev_dataloader,device):
    dev_loss=0.0
    for step,(x,y) in tqdm(enumerate(dev_dataloader)):
        with torch.no_grad():
            loss=model.neg_log_likelihood(x.to(device),y.to(device))
            dev_loss+=loss.item()
    return dev_loss/(step+1)

def train(model,train_dataloader,dev_dataloader,epochs,lr,device):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)

    min_loss,best_epoch=100000.,0 # init of the dev loss
    for epoch in range(epochs):
        total_loss=0.0
        for step,(x,y) in tqdm(enumerate(train_dataloader)):
            m.zero_grad() # clear out the accumulated gradient
            loss=m.neg_log_likelihood(x.to(device),y.to(device))

            loss.backward() # backpropagation of loss
            optimizer.step() # update parameter 

            total_loss+=loss.item()
        train_loss=total_loss/(step+1)

        dev_loss=validate(m,dev_dataloader,device)
        if dev_loss<min_loss:
            min_loss=dev_loss
            best_epoch=epoch
            torch.save(m.state_dict(),'./train_out/bm.ckpt')
        print('epoch {},training loss:{}'.format(epoch,train_loss)+' validation loss:{}'.format(dev_loss))
    print('>>Training done!')

def predict(model,test_dataloader,labels_dict,test_set):
    model.load_state_dict(torch.load('./train_out/bm.ckpt'))
    label_seqs=[]

    idx2label={}
    for key,value in labels_dict.items():
        idx2label[value]=key

    for idx,x in tqdm(enumerate(test_dataloader)):
        #print(x.shape)
        score,idx_seq=model(x)
        label_seq=[idx2label[i] for i in idx_seq][:len(test_set[idx])] # DELETE PAD
        label_seqs.append(label_seq)
    with open('./output/tmp_predict.json','w') as f:
        json.dump(label_seqs,f)
    

def submit(test_file,pred_file):
    with open(pred_file,'r') as f:
        pred=json.load(f)
    
    with open('./output/final_out.txt','w',encoding='utf-8') as f0:

        with open(test_file,'r',encoding='utf-8') as f1:
            for idx,line in enumerate(f1):
                comp_line=line.strip()+'\u0001'+' '.join(pred[idx])
                f0.write(comp_line+'\n')
    print('Done!')

def Main():
    # compute the max len and do the encoding of sequences
    max_len,words_dict,vocab_size,labels_dict,label_size,train_set,dev_set,test_set=get_max_len(('./data/train.conll','./data/dev.conll','./data/final_test.txt'))
    # prepare train dataloader
    train_dataloader=prep_data('train',max_len,train_set)
    # prepare dev dataloader
    dev_dataloader=prep_data('dev',max_len,dev_set)
    # prepare test dataloader
    test_dataloader=prep_data('test',max_len,test_set)
    print('>>data prepared!')
    
    # # build lstm-crf model
    model=BiLSTM_CRF(vocab_size,labels_dict,64,64)
    print('>>model built!')
    # train model
    train(model,train_dataloader,dev_dataloader,epochs=10,lr=1e-2,device='cpu')
    
    # predict
    predict(model,test_dataloader,labels_dict,test_set)

    # submit
    submit('./data/final_test.txt','./output/tmp_predict.json')


if __name__=='__main__':
    Main()



