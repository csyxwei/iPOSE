CUDA_VISIBLE_DEVICES=0 python test_part_syn.py --name partsyn_model --dataset_mode partsplit --gpu_ids 0 \
--dataroot /home/weiyuxiang/datasets/Full_Parts/ --batch_size 32 --part_nc 64 --use_coord --ckpt_iter 540000 --n_support 3 \
--n_att_layers 3