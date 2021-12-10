import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms import RandomCropTransform
from torch.utils.data.dataset import Dataset
from random import choice
from .utils import *

class ACDC(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.do_contrast = args.do_contrast
        self.files = []
        if self.do_contrast:
            # we do not pre-load all data, instead, load data in the get item function
            self.slice_position = []
            self.partition = []
            self.slices = []
            for key in keys:
                frames = subfiles(join(self.data_dir, 'patient_%03d'%key), False, None, ".npy", True)
                for frame in frames:
                    image = np.load(join(self.data_dir, 'patient_%03d'%key, frame))
                    for i in range(image.shape[0]):
                        self.files.append(join(self.data_dir, 'patient_%03d'%key, frame))
                        self.slices.append(i)
                        self.slice_position.append(float(i+1)/image.shape[0])
                        part = image.shape[0] / 4.0
                        if part - int(part) >= 0.5:
                            part = int(part + 1)
                        else:
                            part = int(part)
                        self.partition.append(max(0,min(int(i//part),3)+1))
        else:
            for key in keys:
                frames = subfiles(join(self.data_dir, 'patient_%03d'%key), False, None, ".npy", True)
                for frame in frames:
                    image = np.load(join(self.data_dir, 'patient_%03d'%key, frame))
                    for i in range(image.shape[1]):
                        self.files.append(image[:,i])
        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):
        if not self.do_contrast:
            img = self.files[index][0].astype(np.float32)
            label = self.files[index][1]
            img, label = self.prepare_supervised(img, label)
            return img, label
        else:
            img = np.load(self.files[index]).astype(np.float32)[self.slices[index]]
            img1, img2 = self.prepare_contrast(img)
            return img1, img2, self.slice_position[index], self.partition[index]
            
    # this function for normal supervised training
    def prepare_supervised(self, img, label):
        if self.purpose == 'train':
            # resize image
            img, coord = pad_and_or_crop(img, self.patch_size, mode='random')
            label, _  = pad_and_or_crop(label, self.patch_size, mode='fixed', coords=coord)
            # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
            data_dict = {'data':img[None, None], 'seg':label[None, None]}
            tr_transforms = []
            tr_transforms.append(MirrorTransform((0, 1)))
            tr_transforms.append(RndTransform(SpatialTransform(self.patch_size, list(np.array(self.patch_size)//2),
                                                            True, (100., 350.), (14., 17.),
                                                            True, (0, 2.*np.pi), (-0.000001, 0.00001), (-0.000001, 0.00001),
                                                            True, (0.7, 1.3), 'constant', 0, 3, 'constant', 0, 0,
                                                            random_crop=False), prob=0.67, alternative_transform=RandomCropTransform(self.patch_size)))

            train_transform = Compose(tr_transforms)
            data_dict = train_transform(**data_dict)
            img = data_dict.get('data')[0]
            label = data_dict.get('seg')[0]
            return img, label
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

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default="d:/data/acdc/acdc_contrastive/contrastive/2d/")
    parser.add_argument("--data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/acdc_contrastive/contrastive/2d/")
    parser.add_argument("--patch_size", type=tuple, default=(352, 352))
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--do_contrast", default=True, action='store_true')
    parser.add_argument("--slice_threshold", type=float, default=0.5)
    args = parser.parse_args()


    train_dataset = ACDC(keys=list(range(1,101)), purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    drop_last=False)

    pp = []
    for batch_idx, tup in enumerate(train_dataloader):
        print(f'the {batch_idx}th/{len(train_dataloader)} minibatch...')
        img1, img2, slice_position, partition = tup
        batch_size = img1.shape[0]
        # print(f'batch_size:{batch_size}, slice_position:{slice_position}')
        slice_position = slice_position.contiguous().view(-1, 1)
        mask = (torch.abs(slice_position.T.repeat(batch_size,1) - slice_position.repeat(1,batch_size)) < args.slice_threshold).float()
        # count how many positive pair in each batch
        for i in range(batch_size):
            pp.append(2*mask[i].sum()-1)
    pp = np.asarray(pp)
    pp_mean = np.mean(pp)
    pp_std = np.std(pp)
    print(f'average number of positive pairs mean:{pp_mean}, std:{pp_std}')




