import torch
from torch import nn
import torch.nn.functional as F


class NTMController(nn.Module):

    def __init__(self, input_size, controller_size, output_size, read_data_size):
        super().__init__()
        self.input_size = input_size
        self.controller_size = controller_size
        self.output_size = output_size
        self.read_data_size = read_data_size

        self.controller_net = nn.LSTMCell(input_size, controller_size)
        self.out_net = nn.Linear(read_data_size, output_size)
        # nn.init.xavier_uniform_(self.out_net.weight)
        nn.init.kaiming_uniform_(self.out_net.weight)
        self.h_state = torch.zeros([1, controller_size])
        self.c_state = torch.zeros([1, controller_size])
        # layers to learn bias values for controller state reset
        self.h_bias_fc = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.h_bias_fc.weight)
        self.c_bias_fc = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.c_bias_fc.weight)
        self.reset()

    def forward(self, in_data, prev_reads):
        x = torch.cat([in_data] + prev_reads, dim=-1)
        self.h_state, self.c_state = self.controller_net(
            x, (self.h_state, self.c_state))
        return self.h_state, self.c_state

    def output(self, read_data):
        complete_state = torch.cat([self.h_state] + read_data, dim=-1)
        output = torch.sigmoid(self.out_net(complete_state))
        return output

    def reset(self, batch_size=1):
        in_data = torch.tensor([[0.]])  # dummy input
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1)
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1)

class NTMControllerTwoLSTM(nn.Module):

    def __init__(self, input_size, controller_size, output_size, read_data_size):
        super().__init__()
        self.input_size = input_size
        self.controller_size = controller_size
        self.output_size = output_size
        self.read_data_size = read_data_size

        self.controller_net1 = nn.LSTMCell(input_size, controller_size)
        self.controller_net2 = nn.LSTMCell(controller_size, controller_size)
        self.out_net = nn.Linear(read_data_size, output_size)
        # nn.init.xavier_uniform_(self.out_net.weight)
        nn.init.kaiming_uniform_(self.out_net.weight)
        self.h_state1 = torch.zeros([1, controller_size])
        self.c_state1 = torch.zeros([1, controller_size])
        self.h_state2 = torch.zeros([1, controller_size])
        self.c_state2 = torch.zeros([1, controller_size])
        # layers to learn bias values for controller state reset
        self.h_bias_fc1 = nn.Linear(1, controller_size)
        self.h_bias_fc2 = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.h_bias_fc.weight)
        self.c_bias_fc1 = nn.Linear(1, controller_size)
        self.c_bias_fc2 = nn.Linear(1, controller_size)
        # nn.init.kaiming_uniform_(self.c_bias_fc.weight)
        self.reset()

    def forward(self, in_data, prev_reads):
        x = torch.cat([in_data] + prev_reads, dim=-1)
        self.h_state1, self.c_state1 = self.controller_net1(
            x, (self.h_state1, self.c_state1))

        self.h_state2, self.c_state2 = self.controller_net2(
            self.h_state1, (self.h_state2, self.c_state2))
        return self.h_state2, self.c_state2

    def output(self, read_data):
        complete_state = torch.cat([self.h_state2] + read_data, dim=-1)
        output = torch.sigmoid(self.out_net(complete_state))
        return output

    def reset(self, batch_size=1):
        in_data = torch.tensor([[0.]])  # dummy input
        h_bias1 = self.h_bias_fc1(in_data)
        h_bias2 = self.h_bias_fc2(in_data)
        self.h_state1 = h_bias1.repeat(batch_size, 1)
        self.h_state2 = h_bias2.repeat(batch_size, 1)
        c_bias1 = self.c_bias_fc1(in_data)
        c_bias2 = self.c_bias_fc2(in_data)
        self.c_state1 = c_bias1.repeat(batch_size, 1)
        self.c_state2 = c_bias2.repeat(batch_size, 1)
