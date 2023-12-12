#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:54:17 2023

@author: benjaminheuberger
"""

import time

out_dir = 'out-lyrics-Snoop-Dogg'
eval_interval = 50
eval_iters = 50
#always_save_checkpoint=False 
wandb_log = False # feel free to turn on
wandb_project = 'lyrics'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'lyrics'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 1
gradient_accumulation_steps = 32
max_iters = 600

# finetune at constant LR
#learning_rate = 9e-5
#learning_rate = 3e-5
learning_rate = 2e-4
#learning_rate = 3e-5
#decay_lr = True

# additional 
dropout = 0.1 