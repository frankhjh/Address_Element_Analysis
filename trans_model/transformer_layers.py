import torch.nn as nn
import torch
from trans_model.transformer_sublayers import ScaleDotProductAttention,MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self,dim_model,dim_hid,n_head,dim_k,dim_v,dropout=0.2):
        super(EncoderLayer,self).__init__()
        self.slf_attn=MultiHeadAttention(n_head,dim_model,dim_k,dim_v)
        self.ffn=PositionwiseFeedForward(dim_model,dim_hid,dropout=dropout)
    

    def forward(self,enc_input,slf_attn_mask=None):
        enc_output,enc_slf_attn=self.slf_attn(enc_input,enc_input,enc_input,mask=slf_attn_mask) #mask:Boolean构成的[B,1,L]
        output=self.ffn(enc_output)
        return output,enc_slf_attn
        
class DecoderLayer(nn.Module):
    def __init__(self,dim_model,dim_hid,n_head,dim_k,dim_v,dropout=0.2):
        super(DecoderLayer,self).__init__()
        self.slf_attn=MultiHeadAttention(n_head,dim_model,dim_k,dim_v,dropout=dropout)
        self.enc_attn=MultiHeadAttention(n_head,dim_model,dim_k,dim_v,dropout=dropout)
        self.ffn=PositionwiseFeedForward(dim_model,dim_hid,dropout=dropout)

    def forward(self,enc_output,dec_input,slf_attn_mask=None,dec_enc_attn_mask=None):
        dec_output,dec_slf_attn=self.slf_attn(dec_input,dec_input,dec_input,mask=slf_attn_mask)

        dec_output,dec_enc_attn=self.enc_attn(dec_output,enc_output,enc_output,mask=dec_enc_attn_mask)

        dec_output=self.ffn(dec_output)
        return dec_output,dec_slf_attn,dec_enc_attn
        



