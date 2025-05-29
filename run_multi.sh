export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_floorplan.py  8
