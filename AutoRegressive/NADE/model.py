import torch
import torch.nn as nn

class MaskedLinear(nn.Linear):
  def __init__(self, d, h, bias = True):
    super(MaskedLinear, self).__init__(d, h, bias)
    self.register_buffer('mask', torch.ones(d, 1))
    self.t = torch.arange(d)

  def set_mask(self, i): # starting from zero
    self.mask.copy_data((self.t[:, None] < i).float())

  def forward(self, x):
    return nn.functional.linear(x * self.mask, self.wight, self.bias)

class NADE(nn.Module):
    def __init__(self, visible_size, hidden_size):
        super(NADE, self).__init__()

        self.visible_size = visible_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.randn(hidden_size, visible_size) * 0.01)
        self.V = nn.Parameter(torch.randn(visible_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.c = nn.Parameter(torch.zeros(visible_size))

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.sigmoid(x @ self.W.t() + self.b)
        p_x_given_h = torch.sigmoid(h @ self.V.t() + self.c)
        return p_x_given_h

    def sample(self, n_samples):
        samples = torch.zeros(n_samples, self.visible_size)
        for i in range(self.visible_size):
            p_x_given_h = self.forward(samples)
            samples[:, i] = (torch.rand(n_samples) < p_x_given_h[:, i]).float()
        return samples
