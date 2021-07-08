import torch
import torch.nn as nn
from torch import optim
from trans_model.transformer import Transformer
from loss import my_loss
from data_process import prepare_data
from data_loader import train_dev_loader,test_loader
from tqdm import tqdm
from output.pred_count import done_count

def evaluate(model,metric,data_loader):
    dev_loss=0.0
    for (x,y_input),y_gold in data_loader:
        with torch.no_grad():
            output=model(x,y_input)
            loss=metric(output,y_gold)
            dev_loss+=loss.item()
    return dev_loss


def train(model,epochs,lr,metric,train_data,dev_data):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)
    
    min_loss,best_epoch=100000.0,0
    for epoch in range(epochs):
        total_loss=0.0
        for (x,y_input),y_gold in train_data:
            output=m(x,y_input)
            loss=metric(output,y_gold)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            total_loss+=loss.item()

        dev_loss=evaluate(m,metric,dev_data)
        if dev_loss<min_loss:
            min_loss=dev_loss
            best_epoch=epoch
            torch.save(m.state_dict(),'bm.ckpt')
        print('epoch {},training loss:{}'.format(epoch,total_loss)+' development loss:{}'.format(dev_loss))
    
    print('Training done!\n')

def single_predict(model,src,start_symbol,max_len,exclude_idx): #greedy decoding
    y_pred=torch.ones(1,1).fill_(start_symbol).type_as(src)
    memory=model.encode(src)
    for i in range(max_len):
        out=model.decode(src,y_pred,memory)
        prob=out[:,-1,:]
        #print(prob)
        _,max_idx=prob.topk(4) #NOT CONSIDER 'SOS'/'EOS'/'P'
        p=0
        while max_idx[0][p]>=exclude_idx:
            p+=1
        next_word=max_idx[0][p].item()
        y_pred=torch.cat([y_pred,torch.ones(1,1).fill_(next_word).type_as(src)],dim=1)
    return y_pred

def complete_predict(model,test_data,start_symbol,max_len,exclude_idx,test_set,trans_dict,out_path,start_idx=0):
    sub_pred=[] 
    print(f'{start_idx/500}% prediction task has done!')
    test_data=test_data[start_idx:,:]
    testloader=test_loader(test_data,batch_size=1)
    
    for idx,test_src in tqdm(enumerate(testloader)):
        pred=single_predict(model,test_src,start_symbol,max_len,exclude_idx).tolist()
        for i in range(len(pred[0])):
            pred[0][i]=trans_dict[pred[0][i]]
        sub_pred+=pred

        if idx%100==0:
            with open(out_path,'a',encoding='utf-8') as f:
                for p in sub_pred:
                    f.write(' '.join(p)+'\n')

    return sub_pred       
                

if __name__=='__main__':
    #prepare data
    (train_data,dev_data,test_data),m_src,src_set,test_set=prepare_data('data/train.conll','data/dev.conll','data/final_test.txt',src=True)
    (train_tar,dev_tar),m_tar,tar_set,max_len=prepare_data('data/train.conll','data/dev.conll','data/final_test.txt',src=False)
    
    #build data loader
    train_loader=train_dev_loader(train_data,train_tar,batch_size=32,train=True,Transformer=True)
    dev_loader=train_dev_loader(dev_data,dev_tar,batch_size=32,train=False,Transformer=True)
    #test_loader=test_loader(test_data,batch_size=1) #each time predicts 1 test sample
    
    trans_dict={}
    for key,value in m_tar.items():
        trans_dict[value]=key
    
    #parameters
    tar_size=len(tar_set)+3 #'P'/'SOS'/'EOS'
    src_size=len(src_set)+1
    hidden_size=128
    num_layers=2
    num_heads=4
    dim_k=32
    dim_v=32
    pad_idx=len(src_set)
    mask_idx=len(tar_set)

    epochs=50
    lr=1e-3
    torch.manual_seed(2)
    metric=my_loss(mask_idx)
    model=Transformer(src_size,tar_size,pad_idx,mask_idx,hidden_size,hidden_size,hidden_size,dim_k,dim_v,num_heads,num_layers)

    # print('Start Training...')
    # train(model,epochs,lr,metric,train_loader,dev_loader)

    print('Start Loading Saved Model...')
    model.load_state_dict(torch.load('bm.ckpt'))
    print('Loaded Done!')

    count=done_count('output/tmp_pred.txt')
    complete_predict(model,test_data,len(tar_set)+1,max_len,mask_idx,test_set,trans_dict,'output/tmp_pred.txt',count)

    




        





