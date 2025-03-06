# --> General imports
import os
import numpy as np
import torch
import h5py
import pandas as pd
# --> Torch imports 
import torch
from torch.utils.data import Dataset
import pdb
from numpy.random import randint


class MoSAReDataset(Dataset):
    def __init__(self, feats_dir_wsi, rna_dir, n_tokens, local_wsi_dir, meta_dir,text_local, text_global, missing_percentage, sampling_strategy="random", split="Train"):
        """
        - feats_dir_wsi: str, directory where feat .pt files are stored
        - rna_dir: str, directory where rna_data .csv files are stored
        - n_tokens: int, number of tokens/patches to sample from all features
        - sampling_strategy: str, strategy to sample patches (only "random" available)
        """
        self.feats_dir = feats_dir_wsi
        self.rna_dir = rna_dir
        self.n_tokens = n_tokens
        self.sampling_strategy = sampling_strategy
        self.prototype_dir = local_wsi_dir #TODO: Modification
        self.local_text_dir=text_local
        self.global_text_dir=text_global
        self.rna = pd.read_csv(rna_dir)
        self.meta_dir = pd.read_csv(meta_dir)
        self.mask = get_mask(view_num= 3,alldata_len = self.meta_dir.shape[0] ,missing_rate=missing_percentage) #TODO: Modification

    def __len__(self):
        return self.meta_dir.shape[0]

    def __getitem__(self, idx,masked=True):
        # Fine ID
        slide_id = self.meta_dir.iloc[idx]['filename'].split('.sv')[0]
        # Load features and coords 
        patch_emb = torch.load(os.path.join(self.feats_dir, f"{slide_id}.pt"))
        proto_wsi_local = torch.load(os.path.join(self.prototype_dir, f"{slide_id}.pt")) #TODO: Modification
        # Stardardization WSI
        patch_emb = standardization(patch_emb)
        proto_wsi_local = standardization(proto_wsi_local)
        
        # Define label
        label = self.meta_dir.iloc[idx]['label'] #TODO: Modification
        if label=='LUAD':
            label_sample=1
        elif label=='LUSC':
            label_sample=0    
       
        # WSI 
        patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
        patch_emb_ = patch_emb[patch_indices]
        
        # And an augmentation
        patch_indices_aug = torch.randint(0, patch_emb.size(0), (self.n_tokens,)).tolist() if patch_emb.shape[0] < self.n_tokens else torch.randperm(patch_emb.size(0))[:self.n_tokens].tolist()           
        patch_emb_aug = patch_emb[patch_indices_aug]

        # Load gene expression data 
        rna_data = self.rna[self.rna['index'].str[:15]==slide_id[:12]+'-01'].iloc[:,1:]
        
        # Stardardization
        rna_data = np.array(rna_data.iloc[0].values, dtype=np.float32)  # Convert all values to float32
        rna_data = torch.tensor(rna_data)
        rna_data = standardization(rna_data.unsqueeze(0))
        rna_data = rna_data.squeeze(0)
        
        # Load text data 
        proto_text_local = torch.load(os.path.join(self.local_text_dir, f"{slide_id}.pt")) #TODO: Modification
        proto_text_global = torch.load(os.path.join(self.global_text_dir, f"{slide_id}.pt")) #TODO: Modification
        
        # Stardardization text
        proto_text_local = standardization(proto_text_local)
        proto_text_global = torch.tensor(proto_text_global)
        proto_text_global = standardization(proto_text_global)

        if masked:
            patch_emb_ = patch_emb_*self.mask[0,idx] #TODO: Modification
            proto_wsi_local = proto_wsi_local*self.mask[0,idx] #TODO: Modification
            rna_data = rna_data*self.mask[1,idx] #TODO: Modification
            proto_text_local = proto_text_local*self.mask[2,idx] #TODO: Modification
            proto_text_global = proto_text_global.clone().detach() * self.mask[2, idx]
        
        return patch_emb_, rna_data, proto_wsi_local,label_sample,proto_text_local,proto_text_global,self.mask[0,idx],self.mask[1,idx],self.mask[2,idx] #TODO: Modification

def standardization(a):
    mean_a = torch.mean(a,dim=1)
    std_a = torch.std(a,dim=1)
    n_a = a.sub_(mean_a[:, None]).div_(std_a[:, None])
    return n_a
def get_mask(view_num, alldata_len, missing_rate, split='train'): #TODO: Modification
    nums = np.ones((view_num, alldata_len))
    nums[:, :int(alldata_len * missing_rate)] = 0
    
    for i in range (view_num):
        np.random.shuffle(nums[i])

    count = np.sum(nums, axis=0)
    add=0
    for i in range (alldata_len): # if all of 4 moduls are missing, it does not make  sense, make one of the moduls ==1
        if(count[i]==0):
            nums[randint(0,view_num)][i]=1
            add+=1
    dele=0
    one=0
    count = np.sum(nums, axis=0)
    for i in range (alldata_len): # to keep the rate of missing one equal to missing_rate
        if(add==dele):
            break;
        if(count[i]>1):
            bb=randint(0, view_num )
            nums[bb][i]=0
            dele+=1
            one+=bb

    nums = torch.from_numpy(nums)
    indices = torch.where((nums[0] == 0) & (nums[1] == 0))[0]
    # Split into two halves
    half = len(indices) // 2
    first_half = indices[:half]  # First half of indices
    second_half = indices[half:]  # Second half of indices

    # Set first row for first half indices to 1
    nums[0, first_half] = 1

    # Set second row for second half indices to 1
    nums[1, second_half] = 1


    return nums.to(torch.float32)


def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        feats = hdf5_file['features'][:].squeeze()
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return feats


class SlideDataset(Dataset):
    def __init__(self, features_path, extension='.h5'):
        """
        Args:
            features_path (string): Directory with all the feature files.
        """
        self.features_path = features_path
        self.extension = extension
        self.slide_names = [x for x in os.listdir(features_path) if x.endswith(extension)]
        self.n_slides = len(self.slide_names)

    def __len__(self):
        return self.n_slides

    def __getitem__(self, index):
        slide_id = self.slide_names[index].replace(self.extension, '')
        feature_file = self.slide_names[index]
        feature_path = f"{self.features_path}/{feature_file}"
        if self.extension == '.pt':
            features = torch.load(feature_path)
        elif self.extension == '.h5':
            features = load_h5(feature_path)
        return features, slide_id
