import torch
import torch.nn as nn
from torch import optim
from encoder_model import Attn_Model
from loss import my_loss
from data_prepare import prepare_data
from data_loader import train_dev_loader,test_loader

def evaluate(model,metric,data_loader):
    dev_loss=0.0
    for x,y in data_loader:
        with torch.no_grad():
            output=model(x)
            loss=metric(output,y)
            dev_loss+=loss.item()
    return dev_loss

def train(model,epochs,lr,metric,train_data,dev_data):
    m=model
    optimizer=optim.Adam(m.parameters(),lr=lr)
    
    min_loss,best_epoch=100000.0,0
    for epoch in range(epochs):
        total_loss=0.0
        for x,y in train_data:
            output=m(x)
            loss=metric(output,y)
            
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

def sub_predict(model,test_data,trans_dict):
    print('start predicting...')
    predictions=model(test_data) #shape of output=(,seq_len,output_size)
    print('prediction done!')
    
    top_value,top_idx=predictions.topk(2)
    #shape of top_idx=(,seq_len,1)
    
    #we need to transfer the index back to the tag
    pred_tag_class=[[] for i in range(test_data.size(0))]
    for i in range(top_idx.size(0)):
        for j in range(top_idx.size(1)):
            opt=trans_dict[int(top_idx[i][j][0])]
            if opt=='P':
                opt=trans_dict[int(top_idx[i][j][1])]
            pred_tag_class[i].append(opt)
    
    return pred_tag_class

def predict_full(model,test_dataloader,test_set,trans_dict):
    
    final_pred=[]
    for i,sub in enumerate(test_dataloader):
        sub_pred=sub_predict(model,sub,trans_dict)
        final_pred+=sub_pred
        print(f"{(i+1)*100} test samples done!")
    
    #slice
    for i in range(len(test_set)):
        final_pred[i]=final_pred[i][:len(test_set[i])]
        final_pred[i]=' '.join(final_pred[i])
    
    return final_pred
        
def submit(source_file,target_file,prediction):
    output=[]
    with open(source_file,'r',encoding='utf-8') as f1:
        i=0
        for line in f1:
            line=line.strip()
            line=line+'\u0001'+prediction[i]
            output.append(line)
            i+=1
    f1.close()
    with open(target_file,'w',encoding='utf-8') as f2:
        for item in output:
            f2.write(item+'\n')
    f2.close()

 
if __name__=='__main__':
    (train_data,dev_data,test_data),(train_tar,dev_tar),test_set,token2idx,tc2idx,token_li,tag_class_li=prepare_data('data/train.conll','data/dev.conll','data/final_test.txt')
    #data loader
    train_loader=train_dev_loader(train_data,train_tar,batch_size=32)
    dev_loader=train_dev_loader(dev_data,dev_tar,batch_size=32,train=False)
    test_loader=test_loader(test_data,batch_size=100)
    
    trans_dict={}
    for key,value in tc2idx.items():
        trans_dict[value]=key
    
    #parameters
    output_size=len(tag_class_li)+1
    token_size=len(token_li)+1
    hidden_size=128
    num_layers=2
    num_heads=4
    dim_k=32
    dim_v=32
    pad_idx=len(token_li)
    mask_idx=len(tag_class_li)

    epochs=50
    lr=1e-3
    torch.manual_seed(2)
    metric=my_loss(mask_idx)
    model=Attn_Model(output_size,token_size,hidden_size,num_layers,num_heads,dim_k,dim_v,pad_idx)

    print('Start Training...')
    train(model,epochs,lr,metric,train_loader,dev_loader)

    print('Start Loading Saved Model...')
    model.load_state_dict(torch.load('bm.ckpt'))
    print('Loaded Done!')

    prediction=predict_full(model,test_loader,test_set,trans_dict)

    submit('final_test_cp.txt','胡小白_addr_parsing_runid.txt',prediction)

    







    
