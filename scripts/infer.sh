#!/bin/bash
export CUDA_VISIBLE_DEVICES="0";
# xinlai/LISA-13B-llama2-v0-explanatory
# xinlai/LISA-13B-llama2-v0
version="xinlai/LISA-13B-llama2-v0" 
python chat.py --version=${version}