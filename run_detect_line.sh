#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate tensor_new

# run tensorflow in CPU mode
export CUDA_VISIBLE_DEVICES=""

echo $CUDA_VISIBLE_DEVICES
# run detect line
python -W ignore ./detect_slip_lines.py  \
--model_path ./indent_segmentation/sub_010.0012 \
--H=100 2>&1


