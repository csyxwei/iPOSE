OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train_part_syn.py --name partsyn_model --dataset_mode partsplit --gpu_ids 0 \
--dataroot /home/weiyuxiang/datasets/Full_Parts/ --batch_size 6 --num_epochs 500 --seed 42 --freq_print 1000 --freq_fid 5000 --part_nc 64 --no_EMA --use_coord \
--n_support 3
