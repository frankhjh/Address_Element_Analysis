from utils.dataset_init import Train_Dev_Dataset,Test_Dataset
from torch.utils.data import DataLoader

def train_dev_loader(tensor_x,tensor_y,batch_size,train=True):
    
    dataset=Train_Dev_Dataset(tensor_x,tensor_y)
    if train:
        return DataLoader(dataset,batch_size,shuffle=True)
    return DataLoader(dataset,batch_size)

def test_loader(tensor_x,batch_size):
    
    dataset=Test_Dataset(tensor_x)
    return DataLoader(dataset,batch_size,shuffle=False)

    
