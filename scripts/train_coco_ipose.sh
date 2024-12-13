OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ipose.py --name oasis_coco_partsyn2_bsz32_fewshotnew_n3 --dataset_mode cocoins --gpu_ids 0,1,2,3 \
--dataroot /home/weiyuxiang/datasets/COCO --batch_size 28 --num_epochs 500 --part_nc 32 --use_coord --use_globalD --add_vgg_loss --lambda_vgg 10
