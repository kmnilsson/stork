import torch
import torch.nn.functional as F
from torch.nn import Parameter

from stork.nodes.lif.base import LIFGroup


class BaronigAdLIFGroup(LIFGroup):
    def __init__(self, nb_units, tau_ada=200e-3, **kwargs):
        super().__init__(nb_units, **kwargs)
        self.tau_ada = tau_ada
        self.ada = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dt_ = torch.tensor(time_step, device=device, dtype=dtype)

        a = torch.rand(self.shape, device=device, dtype=dtype, requires_grad=True)
        b = torch.rand(self.shape, device=device, dtype=dtype, requires_grad=True)
        ada_dcy_param = torch.randn(self.shape, device=device, dtype=dtype, requires_grad=True)
        
        self.a = Parameter(a, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)
        self.ada_dcy_param = Parameter(ada_dcy_param, requires_grad=True)
        # Reset state is invoked by configure
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.ada = self.get_state_tensor("ada", state=self.ada)
        self.dcy_ada = torch.exp(-self.dt_ / (F.softplus(self.ada_dcy_param) * self.tau_ada))
        self.scl_ada = 1.0 - self.dcy_ada

    def forward(self):
        new_out, rst = self.get_spike_and_reset(self.mem)

        new_syn = (self.dcy_syn * self.syn +
                   self.input)
        
        new_mem = (self.dcy_mem * self.mem +
                   self.scl_mem * (self.syn - self.ada)) * (1.0 - rst)
        
        new_ada = (self.dcy_ada * self.ada +
                   self.scl_ada * (F.softplus(self.a)*self.mem + F.softplus(self.b)*new_out))
        
        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn
        self.ada = self.states["ada"] = new_ada
