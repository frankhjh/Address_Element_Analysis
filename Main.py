import torch
import torch.nn as nn
from torch import optim
from Attn_model import *
from My_loss import my_loss

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
        print('epoch {},training loss:{}'.format(epoch,total_loss)+' development loss:{}'.format(val_loss))
    
    print('Training done!\n')

def predict(model,test_data):
    predictions=model(test_data) #shape of output=(,seq_len,output_size)
    top_value,top_idx=predictions.topk(1)
    




    
