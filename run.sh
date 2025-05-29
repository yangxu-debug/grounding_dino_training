export CUDA_VISIBLE_DEVICES=3
python3 tools/train.py configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_floorplan.py --work-dir results_orisize