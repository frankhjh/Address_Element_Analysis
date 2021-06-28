import torch
import torch.nn as nn
from trans_model.transformer_encoder import Encoder


class Attn_Model(nn.Module):
    def __init__(self,output_size,token_size,hidden_size,num_layers,num_heads,dim_k,dim_v,pad_idx):
        super(Attn_Model,self).__init__()
        self.encoder=Encoder(token_size,hidden_size,num_layers,num_heads,dim_k,dim_v,hidden_size,hidden_size,pad_idx)
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.out=nn.Linear(self.hidden_size,self.output_size)
        self.softmax=nn.Softmax(dim=2)
    
    def forward(self,x):
        tmp=self.encoder(x)
        out=self.softmax(self.out(tmp))
        return out        