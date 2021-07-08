import torch
import torch.nn as nn
import numpy as np
from trans_model.transformer_layers import EncoderLayer,DecoderLayer

def get_pad_mask(seq,pad_idx):
    return (seq!=pad_idx).unsqueeze(-2)

def seq_mask(seq):
    batch_size,seq_len=seq.size()
    sub_seq_mask = (1 - torch.triu(torch.ones((1,seq_len,seq_len)),diagonal=1)).bool()
    return sub_seq_mask


#定义位置信息
class PositionalEncoding(nn.Module):

    def __init__(self, dim_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        #缓存在内存中，常量
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, dim_hid))

    def _get_sinusoid_encoding_table(self, n_position, dim_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dim_hid) for hid_j in range(dim_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):

    def __init__(self,vocab_size,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,pad_idx,dropout=0.2,n_position=200):
        super(Encoder,self).__init__()
        self.embedding_layer=nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=dim_word_vec,
                                          padding_idx=pad_idx)
        self.positionencode=PositionalEncoding(dim_hid=dim_word_vec,n_position=200)
        self.dropout=nn.Dropout(dropout)
        self.layer_stacks=nn.ModuleList([
            EncoderLayer(dim_model=dim_model,dim_hid=dim_hid,n_head=n_head,dim_k=dim_k,dim_v=dim_v)
          for _ in range(n_layers)])
        self.layer_norm=nn.LayerNorm(dim_model,eps=1e-6)
        self.dim_model=dim_model
        self.pad_idx=pad_idx
    
    def forward(self,x,x_mask):
        
        token_embedd=self.embedding_layer(x)
        token_position_embedd=self.dropout(self.positionencode(token_embedd))
        encode_output=self.layer_norm(token_position_embedd) #shape=[B,L,E]--->(batch_size,seq_len,embed_dim)

        #mask=get_pad_mask(x,self.pad_idx) #shape=[B,1,L]
        for encode_layer in self.layer_stacks:
            encode_output,enc_slf_attn=encode_layer(encode_output,slf_attn_mask=x_mask)
        
        return encode_output



class Decoder(nn.Module):

    def __init__(self,tar_size,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,pad_idx,dropout=0.2,n_position=200):
        super(Decoder,self).__init__()
        self.embedding_layer=nn.Embedding(num_embeddings=tar_size,
                                          embedding_dim=dim_word_vec,
                                          padding_idx=pad_idx) #here the pad_idx is not the same one as encoder
        self.positionencode = PositionalEncoding(dim_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.layer_stacks = nn.ModuleList([
            DecoderLayer(dim_model,dim_hid,n_head,dim_k,dim_v,dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        self.dim_model=dim_model
        self.pad_idx=pad_idx

    
    def forward(self,x,enc_output,x_mask,enc_mask):
        tar_embedd=self.embedding_layer(x)
        tar_position_embedd=self.dropout(self.positionencode(tar_embedd))
        dec_output=self.layer_norm(tar_position_embedd)

        for decoder_layer in self.layer_stacks:
            dec_output,dec_slf_attn,dec_enc_attn=decoder_layer(enc_output,dec_output,slf_attn_mask=x_mask,dec_enc_attn_mask=enc_mask)
        
        return dec_output

class Transformer(nn.Module):
    def __init__(self,n_src_vocab,n_tar_vocab,src_pad_idx,tar_pad_idx,dim_word_vec,dim_model,dim_hid,dim_k,dim_v,n_head,n_layers,dropout=0.2,n_position=200):
        super(Transformer,self).__init__()
        self.src_pad_idx=src_pad_idx
        self.tar_pad_idx=tar_pad_idx

        self.encoder=Encoder(n_src_vocab,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,src_pad_idx,dropout,n_position)
        self.decoder=Decoder(n_tar_vocab,dim_word_vec,n_layers,n_head,dim_k,dim_v,dim_model,dim_hid,tar_pad_idx,dropout,n_position)

        self.out=nn.Linear(dim_model,n_tar_vocab)
        self.softmax=nn.Softmax(dim=2)

    
    def forward(self,src_seq,tar_seq):
        src_mask=get_pad_mask(src_seq,self.src_pad_idx)
        tar_mask=get_pad_mask(tar_seq,self.tar_pad_idx) & seq_mask(tar_seq)

        enc_output=self.encoder(src_seq, src_mask)
        dec_output=self.decoder(tar_seq,enc_output,tar_mask,src_mask)

        output=self.softmax(self.out(dec_output))
        return output
    
    def encode(self,src_seq):
        src_mask=get_pad_mask(src_seq,self.src_pad_idx)
        return self.encoder(src_seq,src_mask)
    
    def decode(self,src_seq,tar_sub_seq,memory):
        src_mask=get_pad_mask(src_seq,self.src_pad_idx)
        tar_mask=get_pad_mask(tar_sub_seq,self.tar_pad_idx) & seq_mask(tar_sub_seq)
        dec_out=self.decoder(tar_sub_seq,memory,tar_mask,src_mask)

        output=self.softmax(self.out(dec_out))
        return output


        
    



        


















        
        
        
