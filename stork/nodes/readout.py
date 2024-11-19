import numpy as np
import torch

import torch.nn.functional as F
from torch.nn import Parameter

from stork.nodes.base import CellGroup


class ReadoutGroup(CellGroup):
    def __init__(self,
                 shape,
                 tau_mem=10e-3,
                 tau_syn=5e-3,
                 weight_scale=1.0,
                 initial_state=-1e-3,
                 stateful=False,
                 learn_tau_mem=False,
                 learn_tau_syn=False,
                 tau_mem_hetero=False,
                 tau_syn_hetero=False):
        super().__init__(shape, stateful=stateful, name="Readout")
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.learn_tau_mem = learn_tau_mem
        self.learn_tau_syn = learn_tau_syn
        self.tau_mem_hetero = tau_mem_hetero
        self.tau_syn_hetero = tau_syn_hetero
        self.store_output_seq = True
        self.initial_state = initial_state
        self.weight_scale = weight_scale
        self.out = None
        self.syn = None
        self.mem_param = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)

        time_step = torch.tensor(self.time_step, device=self.device, dtype=self.dtype)

        self.dcy_mem = torch.exp(-time_step / self.tau_mem)
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = torch.exp(-time_step / self.tau_syn)
        self.scl_syn = (1.0 - self.dcy_syn) * self.weight_scale

        if self.learn_tau_mem:
            if self.tau_mem_hetero:
                mem_param_shape = self.shape
                mem_param = torch.randn(mem_param_shape, device=device, dtype=dtype, requires_grad=True) / 4 + 1
            else:
                mem_param = torch.ones(1, device=device, dtype=dtype)
            self.mem_param = Parameter(mem_param, requires_grad=self.learn_tau_mem)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.learn_tau_mem and self.mem_param is not None:
            time_step = torch.tensor(self.time_step, device=self.device, dtype=self.dtype)
            self.dcy_mem = torch.exp(-time_step / (self.tau_mem * F.softplus(self.mem_param)))
            self.scl_mem = 1.0 - self.dcy_mem
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.syn = self.get_state_tensor("syn", state=self.syn)

    def forward(self):
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = self.dcy_mem * self.out + self.scl_mem * self.syn

        self.out = self.states["out"] = new_mem
        self.syn = self.states["syn"] = new_syn
        # self.out_seq.append(self.out)
