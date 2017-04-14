import math, pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()

	self.linear1 = nn.Linear(num_inputs, 200)
        self.lstm = nn.LSTMCell(200, 128)
        num_outputs = action_space.shape[0]
	# Actor
	self.mu_linear = nn.Linear(128, num_outputs)
	self.sigma_sq_linear = nn.Linear(128, num_outputs)
	# Critic
	self.value_linear = nn.Linear(128, 1)

	# initialize weight
        self.apply(weights_init)
	self.mu_linear.weight.data = normalized_columns_initializer(
				self.mu_linear.weight.data, 0.01)
	self.sigma_sq_linear.weight.data = normalized_columns_initializer(
				self.sigma_sq_linear.weight.data, 0.01)
	self.mu_linear.bias.data.fill_(0)
	self.sigma_sq_linear.bias.data.fill_(0)	

        self.value_linear.weight.data = normalized_columns_initializer(
            			self.value_linear.weight.data, 1.0)
        self.value_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
	x = F.relu(self.linear1(inputs))
	x = x.view(-1, 200)
	hx, cx = self.lstm(x, (hx, cx))
	x = hx
	
	return self.value_linear(x), self.mu_linear(x), self.sigma_sq_linear(x), (hx, cx)
