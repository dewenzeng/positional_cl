import pickle
import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from torch.utils.data.dataset import Dataset
from random import choice
from .utils import *

class CHD(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = []
        with open(os.path.join(self.data_dir, "mean_std.pkl"), 'rb') as f:
            mean_std = pickle.load(f)
        if self.do_contrast:
            # we do not pre-load all data, instead, load data in the get item function
            self.slice_position = []
            self.partition = []
            self.means = []
            self.stds = []
            for key in keys:
                frames = subfiles(join(self.data_dir, 'train', key), False, None, ".npy", True)
                frames.sort()
                i = 0
                for frame in frames:
                    self.files.append(join(self.data_dir, 'train', key, frame))
                    self.means.append(mean_std[key]['mean'])
                    self.stds.append(mean_std[key]['std'])
                    self.slice_position.append(float(i+1)/len(frames))
                    part = len(frames) / 4.0
                    if part - int(part) >= 0.5:
                        part = int(part + 1)
                    else:
                        part = int(part)
                    self.partition.append(max(0,min(int(i//part),3)+1))
                    i = i + 1
        else:
            self.means = []
            self.stds = []
            for key in keys:
                frames = subfiles(join(self.data_dir, 'train', 'ct_'+str(key)), False, None, ".npz", True)
                frames.sort()
                for frame in frames:
                    self.means.append(mean_std['ct_'+str(key)]['mean'])
                    self.stds.append(mean_std['ct_'+str(key)]['std'])
                    self.files.append(join(self.data_dir, 'train', 'ct_'+str(key), frame))
        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):
        if self.do_contrast:
            image = np.load(self.files[index]).astype(np.float32)
            # do preprocessing
            image -= self.means[index]
            image /= self.stds[index]
            img1, img2 = self.prepare_contrast(image)
            return img1, img2, self.slice_position[index], self.partition[index]
        else:
            all_data = np.load(self.files[index])['data']
            img = all_data[0].astype(np.float32)
            img -= self.means[index]
            img /= self.stds[index]
            label = all_data[1].astype(np.float32)
            img, label = self.prepare_supervised(img, label)
            return img, label
            
    # this function for normal supervised training
    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
            # pad image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            # No augmentation is used in the finetuning because augmention could hurt the performance.
            return img[None], label[None]

        else:
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='centre')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            return img[None], label[None]

    # use this function for contrastive learning
    def prepare_contrast(self, img):
        # resize image
        img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data':img[None, None]}
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                            True, (100., 350.), (14., 17.),
                                                            True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                            True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                            random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

        train_transform = Compose(tr_transforms)
        data_dict1 = train_transform(**data_dict)
        img1 = data_dict1.get('data')[0]
        data_dict2 = train_transform(**data_dict)
        img2 = data_dict2.get('data')[0]
        return img1, img2

    def  __len__(self):
        return len(self.files)

def get_split_chd(data_dir, fold, seed=12345):
    # this is seeded, will be identical each time
    all_keys = np.arange(0, 50)
    cases = os.listdir(data_dir)
    cases.sort()
    i = 0
    for case in cases:
      all_keys[i] = int(case[-4:])
      i = i + 1
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = kf.split(all_keys)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_keys = all_keys[train_idx]
        test_keys = all_keys[test_idx]
        if i == fold:
            break
    return train_keys, test_keys

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/data/chd/preprocessed_without_label/")
    parser.add_argument("--patch_size", type=tuple, default=(512, 512))
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.05)
    args = parser.parse_args()

    train_keys = os.listdir(os.path.join(args.data_dir,'train'))
    train_keys.sort()
    train_dataset = CHD(keys=train_keys, purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=30,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    drop_last=False)

    pp = []
    n = 0
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {n}th minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
        # print(f'batch_size:{batch_size}, slice_position:{slice_position}')
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size,1) - slice_position.repeat(1,batch_size)) < args.slice_threshold).float()
        # count how many positive pair in each batch
        for i  in range(mask.shape[0]):
            pp.append(mask[i].sum()-1)
        n = n + 1
        if n > 100:
            break
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'mean:{pp_mean}, std:{pp_std}')