# some examples of training command

# contrastive learning
# train pcl on chd dataset
python train_contrast.py --device cuda:0 --batch_size 32 --epochs 300 --data_dir your_data_dir --lr 0.1 --do_contrast --dataset chd --patch_size 512 512\
--experiment_name contrast_chd_pcl_temp01_thresh01_ --slice_threshold 0.1 --temp 0.1 --initial_filter_size 32 --classes 512 --contrastive_method pcl

# train simclr on chd dataset
python train_contrast.py --device cuda:0 --batch_size 32 --epochs 300 --data_dir your_data_dir --lr 0.1 --do_contrast --dataset chd --patch_size 512 512\
--experiment_name contrast_chd_pcl_temp01_thresh01_ --slice_threshold 0.1 --temp 0.1 --initial_filter_size 32 --classes 512 --contrastive_method simclr

# supervised finetuning
# train from scratch on chd dataset using 40 samples.
python train_supervised.py --device cuda:0 --batch_size 10 --epochs 100 --data_dir your_data_dir --lr 5e-5 --min_lr 5e-6 --dataset chd --patch_size 512 512 \
--experiment_name supervised_chd_scratch_sample_40_ --initial_filter_size 32 --classes 8 --enable_few_data --sampling_k 40

# train from pcl pretrained on chd dataset using 40 samples.
python train_supervised.py --device cuda:0 --batch_size 10 --epochs 100 --data_dir your_data_dir --lr 5e-5 --min_lr 5e-6 --dataset chd --patch_size 512 512 \
--experiment_name supervised_chd_pcl_sample_40_ --initial_filter_size 32 --classes 8 --enable_few_data --sampling_k 40 \
--restart --pretrained_model_path your_pretrained_model_path

# Note: For ACDC and HVSMR, we use initial_filter_size 48 to align with the work one sota https://github.com/MIC-DKFZ/ACDC2017. For CHD and MMWHS, we use initial_filter_size 32
