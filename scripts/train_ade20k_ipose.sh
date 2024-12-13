OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ipose.py --name ipose_ade20k --dataset_mode ade20kins --gpu_ids 0,1,2,3 \
--dataroot /home/weiyuxiang/datasets/ADEChallengeData2016 --batch_size 24 --num_epochs 500 --part_nc 32 --use_coord \
--use_globalD --add_vgg_loss --lambda_vgg 10 --n_support 3  --lr_g 0.0002
