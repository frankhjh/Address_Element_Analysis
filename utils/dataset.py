from torch.utils.data import Dataset

class Train_Dev_Dataset(Dataset):
    def __init__(self,addr,label):
        super(Train_Dev_Dataset,self).__init__()
        self.addr=addr
        self.label=label
        
    def __len__(self):
        return len(self.addr)
        
    def __getitem__(self,idx):
        return self.addr[idx],self.label[idx]
    

class Test_Dataset(Dataset):
    def __init__(self,addr):
        super(Test_Dataset,self).__init__()
        self.addr=addr
    
    def __len__(self):
        return len(self.addr)
        
    def __getitem__(self,idx):
        return self.addr[idx]



