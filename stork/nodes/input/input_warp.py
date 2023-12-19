import numpy as np

from stork.nodes.input.base import InputGroup


class InputWarpGroup(InputGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(self, shape, nb_input_steps, scale=1.0, name="Input"):
        super(InputWarpGroup, self).__init__(shape, name=name)
        self.nb_input_steps = nb_input_steps
        self.scale = scale

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.warp = np.array(np.linspace(0, self.nb_input_steps - 1, nb_steps), dtype=int)

    def forward(self):
        self.out = self.states["out"] = self.scale * self.local_data[:, self.warp[self.clk]]