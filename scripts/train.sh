#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7";

deepspeed --master_port=24999 tools/train.py \
                              --version="models--liuhaotian--llava-llama-2-13b-chat-lightning-preview/snapshots/bcda0227a7f371117a195ef0af88c1616a830520" \
                              --dataset_dir='LISA/dataset' \
                              --vision_pretrained="sam_vit_h_4b8939.pth" \
                              --dataset="sem_seg||vqa||reason_seg" \
                              --sample_rates="9,3,1" \
                              --exp_name="lisa-7b" \
                              --batch_size=1