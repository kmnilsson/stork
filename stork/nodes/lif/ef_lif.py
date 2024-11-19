import numpy as np
import torch

import torch.nn.functional as F
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


class EFLIFGroup(CellGroup):
    def __init__(self,
                 shape,
                 tau_mem=10e-3,
                 tau_syn=5e-3,
                 diff_reset=False,
                 learn_tau_mem=False,
                 learn_tau_syn=False,
                 tau_mem_hetero=False,
                 tau_syn_hetero=False,
                 clamp_mem=False,
                 activation=activations.SuperSpike,
                 dropout_p=0.0,
                 stateful=False,
                 name="EFLIFGroup",
                 regularizers=None,
                 **kwargs):
        """
        Leaky integrate-and-fire neuron with Euler-Forward discretization.
        """
        super().__init__(shape, dropout_p=dropout_p, stateful=stateful,
                         name=name, regularizers=regularizers, **kwargs)
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.spk_nl = activation.apply
        self.diff_reset = diff_reset
        self.learn_tau_mem = learn_tau_mem
        self.learn_tau_syn = learn_tau_syn
        self.tau_mem_hetero = tau_mem_hetero
        self.tau_syn_hetero = tau_syn_hetero
        self.clamp_mem = clamp_mem
        self.mem = None
        self.syn = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.alpha = 1.0 - time_step / self.tau_mem
        self.gamma = 1.0 - time_step / self.tau_syn
        
        if self.learn_tau_mem:
            if self.tau_mem_hetero:
                mem_param_shape = self.shape
            else:
                mem_param_shape = 1
            mem_param = torch.randn(mem_param_shape, device=device, dtype=dtype, requires_grad=True)
            mem_param = mem_param / 4 + 1
            self.mem_param = Parameter(mem_param, requires_grad=self.learn_tau_mem)
        
        if self.learn_tau_syn:
            if self.tau_syn_hetero:
                syn_param_shape = self.shape
            else:
                syn_param_shape = 1
            syn_param = torch.randn(syn_param_shape, device=device, dtype=dtype, requires_grad=True)
            syn_param = syn_param / 4 + 1
            self.syn_param = Parameter(syn_param, requires_grad=self.learn_tau_syn)
            
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.learn_tau_mem:
            self.alpha = 1.0 - self.time_step / (self.tau_mem * F.softplus(self.mem_param))
        if self.learn_tau_syn:
            self.gamma = 1.0 - self.time_step / (self.tau_syn * F.softplus(self.syn_param))
        
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.syn = self.get_state_tensor("syn", state=self.syn)
        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

    def get_spike_and_reset(self, mem):
        mthr = mem - 1.0
        out = self.spk_nl(mthr)

        if self.diff_reset:
            rst = out
        else:
            # if differentiation should not go through reset term, detach it from the computational graph
            rst = out.detach()

        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_syn = self.gamma * self.syn + self.input
        new_mem = (self.alpha * self.mem + (1 - self.alpha) * self.syn) * (1.0 - rst)  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn
