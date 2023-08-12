#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7";
deepspeed --master_port=24999 train_ds.py --version="PATH_TO_LLaVA" \
                                          --dataset_dir='./dataset' \
                                          --vision_pretrained="PATH_TO_SAM" \
                                          --exp_name="lisa-7b" \
                                          --weight='PATH_TO_pytorch_model.bin' \
                                          --eval_only