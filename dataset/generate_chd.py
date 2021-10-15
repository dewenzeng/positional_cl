import SimpleITK as sitk
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import pickle
from collections import OrderedDict

def generate_chd_dataset(data_dir, labeled_save_dir, unlabeled_save_dir):
    results = OrderedDict()
    for i in range(1001,1129):
        if os.path.exists(os.path.join(data_dir,'ct_'+str(i)+'_image.nii.gz')):
            print(f'processing i={i}')
            image_path = os.path.join(data_dir,'ct_'+str(i)+'_image.nii.gz')
            label_path = os.path.join(data_dir,'ct_'+str(i)+'_label.nii.gz')
            image = sitk.ReadImage(image_path)
            label = sitk.ReadImage(label_path)
            # print(f'image spacing:{image.GetSpacing()}')
            image_npy = sitk.GetArrayViewFromImage(image)
            label_npy = sitk.GetArrayViewFromImage(label)
            # convert label
            image_npy_copy = image_npy.copy()
            min_val_1p=np.percentile(image_npy_copy,1)
            max_val_99p=np.percentile(image_npy_copy,99)
            image_npy_copy[image_npy_copy<min_val_1p]=min_val_1p
            image_npy_copy[image_npy_copy>max_val_99p]=max_val_99p
            mean = image_npy_copy.astype(float).mean()
            std = image_npy_copy.astype(float).std()
            label_npy_copy = label_npy.copy()
            label_npy_copy[label_npy_copy>7]=0
            results['ct_'+str(i)] = {}
            results['ct_'+str(i)]['mean'] = mean
            results['ct_'+str(i)]['std'] = std
            # print(f'mean:{mean}, std:{std}')
            # we save the integer version instead of float version to save space. normalization is done on-the-fly.
            # we save one labeled version and one unlabeled version, maybe not the best solution, but ok.
            # you can also add new unlabeled CT data into the unlabeled dataset for contrastive learning.
            for j in range(image_npy.shape[0]):
                tmp_image = image_npy_copy[j,:,:]
                tmp_label = label_npy_copy[j,:,:]
                all_data = np.stack([tmp_image, tmp_label],axis=0)
                maybe_mkdir_p(os.path.join(labeled_save_dir, 'train', 'ct_'+str(i)))
                save_path_image = os.path.join(labeled_save_dir, 'train', 'ct_'+str(i), 'frame'+str.format('%03d'%j))
                np.savez_compressed(save_path_image, data=all_data)
                maybe_mkdir_p(os.path.join(unlabeled_save_dir, 'train', 'ct_'+str(i)))
                save_path_image = os.path.join(unlabeled_save_dir, 'train', 'ct_'+str(i), 'frame'+str.format('%03d'%j))
                np.save(save_path_image, tmp_image)
    with open(os.path.join(labeled_save_dir, "mean_std.pkl"), 'wb') as f:
        pickle.dump(results, f)

    with open(os.path.join(unlabeled_save_dir, "mean_std.pkl"), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-indir", help="folder where the extracted training data is", type=str)
    parser.add_argument("-labeled_outdir", help="folder where to save the data for the 2d network", type=str)
    parser.add_argument("-unlabeled_outdir", help="folder where to save the data for the 2d network", type=str)
    args = parser.parse_args()
    generate_chd_dataset(args.indir, args.labeled_outdir, args.unlabeled_outdir)

# python generate_chd.py -indir /afs/crc.nd.edu/user/d/dzeng2/data/chd/raw_image -labeled_outdir /afs/crc.nd.edu/user/d/dzeng2/data/chd/test/supervised -unlabeled_outdir /afs/crc.nd.edu/user/d/dzeng2/data/chd/test/contrastive