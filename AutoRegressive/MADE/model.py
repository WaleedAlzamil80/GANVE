import torch
import torch.nn as nn



class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.Tensor(mask))

    def forward(self, input):
        return nn.functional.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.fc1 = MaskedLinear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([MaskedLinear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.fc_out = MaskedLinear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.create_masks()

    def create_masks(self):
        input_order = torch.arange(self.input_dim)
        hidden_order = torch.arange(self.hidden_dim)

        hi = (input_order[None, :] < hidden_order[:, None]).float()  # hidden by input
        hh = (hidden_order[None, :] <= hidden_order[:, None]).float()  # hidden by hidden
        ih = (hidden_order[None, :] <= input_order[:, None]).float()  # input by hidden

        self.fc1.set_mask(hi)
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_layer.set_mask(hh)
        self.fc_out.set_mask(ih)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        return self.sigmoid(self.fc_out(x))






