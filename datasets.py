from torch.utils.data import Dataset


class RNAdataset(Dataset):
    def __init__( self , root_dir, max_pointcloud_size=-1):
        #make regular pytorch dataset here        
        pass

    def __getitem__(self , index):
         if self.transform:
                self.x=torch.from_numpy(self.x)
         return self.x
        

    def __len__(self):
        return len(self.x)