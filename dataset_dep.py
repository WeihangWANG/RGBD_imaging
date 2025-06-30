from random import seed
import torch.utils.data as data
import torch,pdb
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.rgb_data = hf.get("rgb")
        self.dep_data = hf.get("depth")
        self.label = hf.get("gt")
        self.amp_data = hf.get("amp")
        self.corr_data = hf.get("corr")

    def __getitem__(self, index):
        return torch.from_numpy(self.rgb_data[index,:,:,:]).float(), torch.from_numpy(self.dep_data[index,:,:,:]).float(), torch.from_numpy(self.label[index,:,:,:]).float(), torch.from_numpy(self.amp_data[index,:,:,:]).float(), torch.from_numpy(self.corr_data[index,:,:,:]).float()

    def __len__(self):
        return self.rgb_data.shape[0]
