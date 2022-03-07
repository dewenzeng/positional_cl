import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from torch.utils.data.dataset import Dataset
from random import choice
from .utils import *

class HVSMR(Dataset):

    def __init__(self, keys, purpose, args):
        self.data_dir = args.data_dir
        self.patch_size = args.patch_size
        self.purpose = purpose
        self.classes = args.classes
        self.files = []
        for key in keys:
            frames = subfiles(os.path.join(self.data_dir, 'patient_%d'%key), False, None, ".npy", True)
            frames.sort()
            for frame in frames:
                self.files.append(os.path.join(self.data_dir, 'patient_%d'%key, frame))
        print(f'dataset length: {len(self.files)}')

    def __getitem__(self, index):
        img = np.load(self.files[index])[0].astype(np.float32)
        label = np.load(self.files[index])[1].astype(np.float32)
        img, label = self.prepare_supervised(img, label)
        # print(f'finish transform {self.files[index]}')
        return img, label
            
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

    def  __len__(self):
        return len(self.files)

if __name__ == "__main__":
    import argparse 
    from torch.autograd import Variable
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="d:/data/hvsmr/preprocessed")
    # parser.add_argument("--data_dir", type=str, default="/afs/crc.nd.edu/user/d/dzeng2/data/hvsmr/preprocessed")
    parser.add_argument("--patch_size", type=tuple, default=(320, 320))
    parser.add_argument("--classes", type=int, default=4)
    args = parser.parse_args()

    all_keys = np.arange(0, 10)
    train_dataset = HVSMR(keys=all_keys, purpose='train', args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=5,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    drop_last=False)

    for batch_idx, tup in enumerate(train_dataloader):
        image, label = tup
        print(f'image shape:{image.shape}')
        plt.figure(1)
        img_grid = torchvision.utils.make_grid(image)
        matplotlib_imshow(img_grid, one_channel=False)
        plt.figure(2)
        img_grid = torchvision.utils.make_grid(label)
        matplotlib_imshow(img_grid, one_channel=False)
        plt.show()
        break




