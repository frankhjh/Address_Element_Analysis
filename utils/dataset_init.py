from torch.utils.data import Dataset

class Train_Dev_Dataset(Dataset):
    def __init__(self,src,tar):
        super(Train_Dev_Dataset,self).__init__()
        self.src=src
        self.tar=tar
        
    def __len__(self):
        return len(self.src)
        
    def __getitem__(self,idx):
        return self.src[idx],self.tar[idx]
    
class Train_Dev_Dataset_Trans(Dataset):
    def __init__(self,src,tar):
        super(Train_Dev_Dataset_Trans,self).__init__()
        self.src=src
        self.tar_input=self.tar_shift(tar)[0]
        self.tar_gold=self.tar_shift(tar)[1]
    
    def tar_shift(self,target_seq): #size=[batch_size,seq_len]
        tar_input=target_seq[:,:-1]
        tar_gold=target_seq[:,1:]
        return tar_input,tar_gold
        
    def __len__(self):
        return len(self.src)
        
    def __getitem__(self,idx):
        return (self.src[idx],self.tar_input[idx]),self.tar_gold[idx]

class Test_Dataset(Dataset):
    def __init__(self,src):
        super(Test_Dataset,self).__init__()
        self.src=src
    
    def __len__(self):
        return len(self.src)
        
    def __getitem__(self,idx):
        return self.src[idx]



