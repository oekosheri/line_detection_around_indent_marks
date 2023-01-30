#!/usr/local_rwth/bin/zsh

# run training with the given input args
python -W ignore tag_program  \
--global_batch_size=tag_batch \
--lr=tag_lr \
--count=tag_count \
--augment=tag_aug \
--epoch=tag_epoch 2>&1
