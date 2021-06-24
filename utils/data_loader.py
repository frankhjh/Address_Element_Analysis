from torch.utils.data import Dataset,DataLoader
class Corpus_Dataset(Dataset):
    def __init__(self,addr,elem):
        super(Corpus_Dataset,self).__init__()
        self.addr=addr
        self.elem=elem
    
    def __len__(self):
        return len(self.addr)
        
    def __getitem__(self,idx):
        return self.addr[idx],self.elem[idx] 