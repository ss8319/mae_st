Step 1, pretrain:

$env:PYTHONPATH=".."; & C:\Users\User\miniconda3\python.exe run_pretrain.py --path_to_data_dir demo\data --output_dir .\output_pretrain --device cpu --model mae_vit_large_patch16 --batch_size 1 --epochs 1 --num_frames 16 --t_patch_size 2 --pred_t_dim 8 --decoder_embed_dim 512 --decoder_depth 4 --repeat_aug 4 --sampling_rate 4 --mask_ratio 0.9 --norm_pix_loss --blr 1.6e-3 --warmup_epochs 5 --clip_grad 0.02 --num_workers 0

Step 2, fine-tune from the pretrained weights

$env:PYTHONPATH=".."; & C:\Users\User\miniconda3\python.exe run_finetune.py --path_to_data_dir demo\data --finetune .\output_pretrain\checkpoint-00000.pth --output_dir .\output_finetune --device cpu --model vit_large_patch16 --num_frames 16 --t_patch_size 2 --cls_embed --sep_pos_embed --num_classes 400 --batch_size 1 --repeat_aug 1 --epochs 1 --num_workers 0 --dist_eval
