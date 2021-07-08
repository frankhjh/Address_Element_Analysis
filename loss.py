import torch
from torch import nn

class my_loss(nn.Module):
    def __init__(self,mask_idx):
        super(my_loss,self).__init__()
        self.mask_idx=mask_idx
    
    def forward(self,output,target):
        #output:(batch_size,seq_len,output_size)
        #target:(batch_size,seq_len)
        loss=0
        count=0
        batch_size=target.size(0)
        seq_len=target.size(1)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if target[i][j]!=self.mask_idx:
                    loss+=-torch.log(output[i][j][int(target[i][j])])
                    count+=1
        
        return loss/count    
