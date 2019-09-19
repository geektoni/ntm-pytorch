import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
from torch import nn

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from ntm.args import get_parser

def generate_target_original_plots(iteration, task_params, model_path, image_output):

    dataset = PrioritySort(task_params)
    criterion = nn.BCELoss()

    ntm = NTM(input_size=task_params['seq_width'] + 1,
              output_size=task_params['seq_width'],
              controller_size=task_params['controller_size'],
              memory_units=task_params['memory_units'],
              memory_unit_size=task_params['memory_unit_size'],
              num_heads=task_params['num_heads'],
              save_weigths=True,
              multi_layer_controller=task_params['multi_layer_controller'])

    ntm.load_state_dict(torch.load(model_path))

    # -----------------------------------------------------------------------------
    # --- evaluation
    # -----------------------------------------------------------------------------
    ntm.reset()
    data = dataset[0]  # 0 is a dummy index
    input, target = data['input'], data['target']
    out = torch.zeros(target.size())

    # -----------------------------------------------------------------------------
    # loop for other tasks
    # -----------------------------------------------------------------------------
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        ntm(in_data)

    # passing zero vector as the input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
        out[i] = ntm(in_data)

    loss = criterion(out, target)

    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(221)

    ax1.set_title("Result")
    ax2.set_title("Target")

    sns.heatmap(binary_output, ax=ax1, vmin=0, vmax=1, linewidths=.5, cbar=False, square=True )
    sns.heatmap(target, ax=ax2, vmin=0, vmax=1, linewidths=.5, cbar=False, square=True)

    plt.savefig(image_output+"/priority_sort_{}_{}_{}_{}_{}_{}_{}_{}_{}_image_{}.png".format(
        task_params['seq_width'] + 1,
        task_params['seq_width'],
        task_params['controller_size'],
        task_params['memory_units'],
        task_params['memory_unit_size'],
        task_params['num_heads'],
        task_params['uniform'],
        task_params['random_distr'],
        task_params['multi_layer_controller'],
        iteration
    ))

    fig = plt.figure(figsize=(15,6))
    ax1_2 = fig.add_subplot(211)
    ax2_2 = fig.add_subplot(212)
    ax1_2.set_title("Read Weigths")
    ax2_2.set_title("Write Weights")

    sns.heatmap(ntm.all_read_w, ax=ax1_2, linewidths=.01, square=True)
    sns.heatmap(ntm.all_write_w, ax=ax2_2, linewidths=.01, square=True)

    plt.tight_layout()
    plt.savefig(image_output+"/priority_sort_{}_{}_{}_{}_{}_{}_{}_{}_{}_weigths_{}.png".format(
        task_params['seq_width'] + 1,
        task_params['seq_width'],
        task_params['controller_size'],
        task_params['memory_units'],
        task_params['memory_unit_size'],
        task_params['num_heads'],
        task_params['uniform'],
        task_params['random_distr'],
        task_params['multi_layer_controller'],
        iteration
    ), dpi=250)

    # ---logging---
    print('[*] Checkpoint Loss: %.2f\tError in bits per sequence: %.2f' % (loss, error))
