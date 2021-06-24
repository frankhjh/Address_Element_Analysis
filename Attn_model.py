import torch
import torch.nn as nn

class Multi_Head_Attention(nn.Module):
    
    def __init__(self,num_heads,hidden_size,output_size,token_size,max_len):
        super(Multi_Head_Attention,self).__init__()
        self.num_heads=num_heads
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.token_size=token_size
        self.max_len=max_len
        #self.dropout_rate=0.5
        self.embedding_layer=nn.Embedding(num_embeddings=token_size,
                                          embedding_dim=hidden_size,
                                          padding_idx=None)
        
        self.projection_dim=self.hidden_size//self.num_heads #每个head产生的embedding的dim
        self.query=[nn.Linear(self.hidden_size,self.projection_dim) for i in range(self.num_heads)]
        self.key=[nn.Linear(self.hidden_size,self.projection_dim) for i in range(self.num_heads)]
        self.value=[nn.Linear(self.hidden_size,self.projection_dim) for i in range(self.num_heads)]
    
    def attention(self,q,k,v):
        score=torch.matmul(q,k.transpose(-1,-2))
        scale=q.size(-1)**0.5
        score_scaled=score/scale 
        weights=nn.Softmax(dim=-1)(score_scaled)
        
        output=torch.matmul(weights,v)
        return output
    
    def forward(self,x):

        attens=[self.attention(self.query[i](x),self.key[i](x),self.value[i](x))
                for i in range(self.num_heads)] 
        output=torch.cat(attens,dim=-1) 
        return output


class Token_Position_embedd(nn.Module): #token embedding+position embedding
    def __init__(self,hidden_size,token_size,max_len):
        super(Token_Position_embedd,self).__init__()
        self.hidden_size=hidden_size
        self.token_size=token_size
        self.max_len=max_len
        
        self.token_embedd=nn.Embedding(num_embeddings=self.token_size,
                                       embedding_dim=self.hidden_size,
                                       padding_idx=None)
        self.position_embedd=nn.Embedding(num_embeddings=self.max_len,
                                         embedding_dim=self.hidden_size,
                                         padding_idx=None)
        
    def forward(self,x):
        batch_size=x.shape[0]
        positions=torch.tensor([[i for i in range(self.max_len)] for i in range(batch_size)])
        
        positions_embedd=self.position_embedd(positions)
        x_embedd=self.token_embedd(x)
        return x_embedd+positions_embedd
    

class Attn_Model(nn.Module):
    def __init__(self,num_heads,hidden_size,token_size,output_size,max_len):
        super(Attn_Model,self).__init__()
        self.input_embedd=Token_Position_embedd(hidden_size,token_size,max_len)
        self.attention=Multi_Head_Attention(num_heads,hidden_size,output_size,token_size,max_len)
        self.hidden_size=hidden_size
        self.max_len=max_len
        self.output_size=output_size
        #self.dropout_rate=0.2
        self.out=nn.Linear(self.hidden_size,self.output_size)
        self.softmax=nn.Softmax(dim=2)
    
    
    def forward(self,x):
        embedd_x=self.input_embedd(x) #[batch_size,seq_len,hidden_size]
        attn_x=self.attention(embedd_x) #[batch_size,seq_len,hidden_size]
        output=self.softmax(self.out(attn_x))
        
        return output