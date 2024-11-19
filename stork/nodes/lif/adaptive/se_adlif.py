import torch
import torch.nn.functional as F
from torch.nn import Parameter

from stork.nodes.lif.ef_lif import EFLIFGroup


class SEAdLIFGroup(EFLIFGroup):
    """adLIF with symplectic Euler discretization (Baronig et al., 2024).
    """
    def __init__(self, nb_units, tau_ada=200e-3, **kwargs):
        super().__init__(nb_units, **kwargs)
        self.tau_ada = tau_ada
        self.ada = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dt_ = torch.tensor(time_step, device=device, dtype=dtype)

        a = torch.rand(self.shape, device=device, dtype=dtype, requires_grad=True)
        b = torch.rand(self.shape, device=device, dtype=dtype, requires_grad=True)
        ada_param = torch.randn(self.shape, device=device, dtype=dtype, requires_grad=True)
        
        self.a = Parameter(a, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)
        self.ada_param = Parameter(ada_param, requires_grad=True)
        # Reset state is invoked by configure
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.ada = self.get_state_tensor("ada", state=self.ada)
        self.beta = 1.0 - self.dt_ / (self.tau_ada * F.softplus(self.ada_param))

    def forward(self):
        #new_syn = (self.gamma * self.syn + self.input)
        
        new_mem = (self.alpha * self.mem +
                   (1 - self.alpha) * (self.input - self.ada))
                   #(1 - self.alpha) * (self.syn - self.ada))
        
        new_out, rst = self.get_spike_and_reset(self.mem)
        
        new_mem *= (1.0 - rst)
        
        new_ada = (self.beta * self.ada +
                   (1 - self.beta) * (F.softplus(self.a)*new_mem + F.softplus(self.b)*new_out))
        
        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        #self.syn = self.states["syn"] = new_syn
        self.ada = self.states["ada"] = new_ada
