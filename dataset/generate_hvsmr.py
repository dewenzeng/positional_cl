import SimpleITK as sitk
import os
import numpy as np
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
        min_val_1p=np.percentile(image,1)
        max_val_99p=np.percentile(image,99)
        image[image<min_val_1p]=min_val_1p
        image[image>max_val_99p]=max_val_99p
        image -= image.mean()
        image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image

def generate_hvsmr_dataset(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(10):
        image_path = os.path.join(data_dir, 'Training_dataset_sx_cropped', 'training_sa_crop_pat'+str(i)+'.nii.gz')
        label_path = os.path.join(data_dir, 'Ground_truth', 'training_sa_crop_pat'+str(i)+'-label.nii.gz')
        itk_image = sitk.ReadImage(image_path)
        itk_label = sitk.ReadImage(label_path)
        print(f'image spacing:{itk_image.GetSpacing()}')
        print(f'original image size:{itk_image.GetSize()}')
        image = preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.7, 0.7), keep_z_spacing=True)
        label = preprocess_image(itk_label, is_seg=True, spacing_target=(1, 0.7, 0.7), keep_z_spacing=True)
        print(f'resized image size:{image.shape}')
        if not os.path.exists(os.path.join(save_dir, 'patient_'+str(i))):
            os.mkdir(os.path.join(save_dir, 'patient_'+str(i)))
        for j in range(image.shape[0]):
            tmp_image = image[j,:,:]
            tmp_label = label[j,:,:]
            save_path_image = os.path.join(save_dir, 'patient_'+str(i), 'frame_%03d'%j)
            all_data = np.stack([tmp_image, tmp_label],axis=0)
            np.save(save_path_image, all_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-indir", help="folder where the extracted training data is", type=str, default='d:/data/hvsmr/')
    parser.add_argument("-labeled_outdir", help="folder where to save the data for the 2d network", type=str, default='d:/data/hvsmr/test')
    args = parser.parse_args()
    generate_hvsmr_dataset(args.indir, args.labeled_outdir)
