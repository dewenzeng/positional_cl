import SimpleITK as sitk
import os
import numpy as np
from utils import *
import pickle
from collections import OrderedDict
from skimage.transform import resize

def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res

def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5), keep_z_spacing=False):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)
    if keep_z_spacing:
        spacing_target = list(spacing_target)
        spacing_target[0] = spacing[0]
    if not is_seg:
        order_img = 3
        if not keep_z_spacing:
            order_img = 1
        image = resize_image(image, spacing, spacing_target, order=order_img).astype(np.float32)
        # image -= image.mean()
        # image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image

def generate_mmwhs_dataset(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'supervised')):
        os.mkdir(os.path.join(save_dir, 'supervised'))
    i = 1001
    results = OrderedDict()
    for j in range(20):
        print(f'processing ct_train_{i}...')
        image_path = os.path.join(data_dir,'ct_train_'+str(i)+'_image.nii.gz')
        label_path = os.path.join(data_dir,'ct_train_'+str(i)+'_label.nii.gz')
        itk_image = sitk.ReadImage(image_path)
        itk_label = sitk.ReadImage(label_path)
        label_npy = sitk.GetArrayViewFromImage(itk_label)
        label_npy_copy = label_npy.copy()
        label_npy_copy[label_npy_copy==205]=1
        label_npy_copy[label_npy_copy==420]=2
        label_npy_copy[label_npy_copy==500]=3
        label_npy_copy[label_npy_copy==550]=4
        label_npy_copy[label_npy_copy==600]=5
        label_npy_copy[label_npy_copy==820]=6
        label_npy_copy[label_npy_copy==850]=7
        label_npy_copy[label_npy_copy>100]=0
        itk_label_copy = sitk.GetImageFromArray(label_npy_copy)
        itk_label_copy.SetSpacing(itk_label.GetSpacing())
        itk_label_copy.SetDirection(itk_label.GetDirection())
        print(f'image spacing:{itk_image.GetSpacing()}')
        print(f'original image size:{itk_image.GetSize()}')
        image_npy = preprocess_image(itk_image, is_seg=False, spacing_target=(1, 1.0, 1.0), keep_z_spacing=True)
        label_npy = preprocess_image(itk_label_copy, is_seg=True, spacing_target=(1, 1.0, 1.0), keep_z_spacing=True)
        print(f'resized image size:{image_npy.shape}')
        # convert label
        image_npy_copy = image_npy.copy().astype(np.int16)
        # remove the pixels that are too smaller can increase the image contrast
        min_val_1p=np.percentile(image_npy_copy,1)
        max_val_99p=np.percentile(image_npy_copy,99)
        image_npy_copy[image_npy_copy<min_val_1p]=min_val_1p
        image_npy_copy[image_npy_copy>max_val_99p]=max_val_99p
        image_npy_copy[image_npy_copy<-2000]=-1110
        mean = image_npy_copy.astype(float).mean()
        std = image_npy_copy.astype(float).std()
        results['ct_train_'+str(i)] = {}
        results['ct_train_'+str(i)]['mean'] = mean
        results['ct_train_'+str(i)]['std'] = std
        print(f'after label_npy unique:{np.unique(label_npy)}')
        for n in range(image_npy.shape[0]):
            tmp_image = image_npy_copy[n,:,:]
            tmp_label = label_npy[n,:,:]
            all_data = np.stack([tmp_image, tmp_label],axis=0)
            if not os.path.exists(os.path.join(save_dir, 'supervised', 'ct_train_'+str(i))):
                os.mkdir(os.path.join(save_dir, 'supervised', 'ct_train_'+str(i)))
            save_path_image = os.path.join(save_dir, 'supervised', 'ct_train_'+str(i), 'frame'+str.format('%03d'%n))
            np.savez_compressed(save_path_image, data=all_data)
        i = i + 1

    with open(os.path.join(save_dir, "mean_std.pkl"), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-indir", help="folder where the extracted training data is", type=str, default='d:/data/mmwhs/ct_train1')
    parser.add_argument("-labeled_outdir", help="folder where to save the data for the 2d network", type=str, default='d:/data/mmwhs/test')
    args = parser.parse_args()
    generate_mmwhs_dataset(args.indir, args.labeled_outdir)

# example
# python generate_mmwhs.py -indir /afs/crc.nd.edu/user/d/dzeng2/data/mmwhs/ct/raw_data -labeled_outdir /afs/crc.nd.edu/user/d/dzeng2/data/mmwhs/ct/test
