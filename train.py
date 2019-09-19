#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import numpy as np
import os

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

from utils import *

torch.set_num_threads(2)

args = get_parser().parse_args()

print(args)

# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------
#
#args.task_json = 'ntm/tasks/copy.json'
#args.task_json = 'ntm/tasks/repeatcopy.json'
#args.task_json = 'ntm/tasks/associative.json'
#args.task_json = 'ntm/tasks/ngram.json'
#args.task_json = 'ntm/tasks/prioritysort.json'

task_params = json.load(open(args.task_json))
print(task_params)

dataset = PrioritySort(task_params)
#dataset = CopyDataset(task_params)
#dataset = RepeatCopyDataset(task_params)
#dataset = AssociativeDataset(task_params)
#dataset = NGram(task_params)
#dataset = PrioritySort(task_params)

saved_model_name = "priority_sort_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(
    task_params['seq_width'] + 1,
    task_params['seq_width'],
    task_params['controller_size'],
    task_params['memory_units'],
    task_params['memory_unit_size'],
    task_params['num_heads'],
    task_params['uniform'],
    task_params['random_distr'],
    task_params['multi_layer_controller']
)

# Output directory for tensorboard
configure(args.tb_dir+"/priority_sort_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
    task_params['seq_width'] + 1,
    task_params['seq_width'],
    task_params['controller_size'],
    task_params['memory_units'],
    task_params['memory_unit_size'],
    task_params['num_heads'],
    task_params['uniform'],
    task_params['random_distr'],
    task_params['multi_layer_controller']
))


"""
For the Copy task, input_size: seq_width + 2, output_size: seq_width
For the RepeatCopy task, input_size: seq_width + 2, output_size: seq_width + 1
For the Associative task, input_size: seq_width + 2, output_size: seq_width
For the NGram task, input_size: 1, output_size: 1
For the Priority Sort task, input_size: seq_width + 1, output_size: seq_width
"""
ntm = NTM(input_size=task_params['seq_width'] + 1,
          output_size=task_params['seq_width'],
          controller_size=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'],
          multi_layer_controller=task_params['multi_layer_controller'])

criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)
'''
optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                       betas=(args.beta1, args.beta2))
'''

'''
args.saved_model = 'saved_model_copy.pt'
args.saved_model = 'saved_model_repeatcopy.pt'
args.saved_model = 'saved_model_associative.pt'
args.saved_model = 'saved_model_ngram.pt'
args.saved_model = 'saved_model_prioritysort.pt'
'''

# Path for the model
PATH = args.output_dir+"/priority_sort_{}_{}_{}_{}_{}_{}_{}_{}_{}/".format(
    task_params['seq_width'] + 1,
    task_params['seq_width'],
    task_params['controller_size'],
    task_params['memory_units'],
    task_params['memory_unit_size'],
    task_params['num_heads'],
    task_params['uniform'],
    task_params['random_distr'],
    task_params['multi_layer_controller']
)
# Check if the directory exists. Create it otherwise
if not os.path.isdir(PATH):
    os.makedirs(PATH)
    os.makedirs(PATH+"/images")

bare_path = PATH
PATH = PATH+saved_model_name

# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.reset()

    data = dataset[iter]
    input, target = data['input'], data['target']
    out = torch.zeros(target.size())

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # loop for NGram task
    # -------------------------------------------------------------------------
    '''
    for i in range(task_params['seq_len'] - 1):
        in_data = input[i].view(1, -1)
        ntm(in_data)
        target_data = torch.zeros([1]).view(1, -1)
        out[i] = ntm(target_data)
    '''
    # -------------------------------------------------------------------------

    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()

    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())

    # ---logging---
    if iter % 100 == 0 and iter != 0:
        print('[*] Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        log_value('train_loss', np.mean(losses), iter)
        log_value('bit_error_per_sequence', np.mean(errors), iter)
        losses = []
        errors = []

    # ---checkpoint---
    if iter % args.checkpoint == 0 and iter != 0:
        print ('[*] Creating a checkpoint:')
        torch.save(ntm.state_dict(), PATH+".checkpoint_{}".format(iter))

        # Save an image with
        generate_target_original_plots(iter, task_params, PATH+".checkpoint_{}".format(iter), bare_path+"/images")

# ---saving the model---
torch.save(ntm.state_dict(), PATH)
