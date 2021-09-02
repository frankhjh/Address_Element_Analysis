from torch.utils.data import Dataset

class Train_Dev_Dataset(Dataset):
    def __init__(self,addrs,labels):
        super(Train_Dev_Dataset,self).__init__()
        self.addrs=addrs
        self.labels=labels
        
    def __len__(self):
        return len(self.addrs)
        
    def __getitem__(self,idx):
        return self.addrs[idx],self.labels[idx]

class Test_Dataset(Dataset):
    def __init__(self,addrs):
        super(Test_Dataset,self).__init__()
        self.addrs=addrs
    
    def __len__(self):
        return len(self.addrs)
    
    def __getitem__(self,idx):
        return self.addrs[idx],self.labels[idx]
        

