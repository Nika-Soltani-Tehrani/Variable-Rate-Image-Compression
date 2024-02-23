import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedEncoder(nn.Module):
    """
    FC Encoder composed by 3 512-units fully-connected layers
    """
    def __init__(self, coded_size, patch_size):
        super(FullyConnectedEncoder, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size

        self.fc1 = nn.Linear(3 * patch_size * patch_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.w_bin = nn.Linear(512, self.coded_size)

    def forward(self, x):
        """
        :param x: image typically a 8x8@3 patch image
        :return:
        """
        # Flatten the input
        x = x.view(-1, 3 * self.patch_size ** 2)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.w_bin(x))
        x = torch.sign(x)
        return x


class FullyConnectedDecoder(nn.Module):
    """
    FC Decoder composed by 3 512-units fully-connected layers
    """
    def __init__(self, coded_size, patch_size):
        super(FullyConnectedDecoder, self).__init__()
        self.patch_size = patch_size
        self.coded_size = coded_size

        self.fc1 = nn.Linear(coded_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.last_layer = nn.Linear(512, patch_size * patch_size * 3)

    def forward(self, x):
        """
        :param x: encoded features
        :return:
        """
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.last_layer(x)  # shape: [4, 192]
        x = x.view(-1, 3, self.patch_size, self.patch_size)  # shape: [4, 3, 8, 8]
        return x


class CoreFC(nn.Module):

    def __init__(self, coded_size, patch_size):
        super(CoreFC, self).__init__()

        self.fc_encoder = FullyConnectedEncoder(coded_size, patch_size)
        self.fc_decoder = FullyConnectedDecoder(coded_size, patch_size)

    def forward(self, x, num_pass=0):
        bits = self.fc_encoder(x)
        out = self.fc_decoder(bits)
        return out


class ResidualFullyConnectedNetwork(nn.Module):

    def __init__(self, coded_size=4, patch_size=8, repetition=16):
        super(ResidualFullyConnectedNetwork, self).__init__()
        self.repetition = repetition

        self.encoders = nn.ModuleList([FullyConnectedEncoder(coded_size, patch_size) for i in range(repetition)])
        self.decoders = nn.ModuleList([FullyConnectedDecoder(coded_size, patch_size) for i in range(repetition)])

    def forward(self, input_patch, pass_num):

        out_bits = self.encoders[pass_num](input_patch)
        output_patch = self.decoders[pass_num](out_bits)

        residual_patch = input_patch - output_patch
        return residual_patch

    def sample(self, input_patch):

        outputs = []
        for pass_num in range(self.repetition):
            out_bits = self.encoders[pass_num](input_patch)
            output_patch = self.decoders[pass_num](out_bits)
            outputs.append(output_patch)

            input_patch = input_patch - output_patch

        reconstructed_patch = sum(outputs)
        return reconstructed_patch
