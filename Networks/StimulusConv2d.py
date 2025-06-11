# Import packages
import torch.nn as nn

from _utils_.OpenFaceInstance import OpenFaceInstance
from Networks.StimulusConv1d import StimulusConv1d


# Class
class StimulusConv2d(StimulusConv1d):

    def __init__(self, params=None, n_classes=2, separated_inputs=True):
        super(StimulusConv2d, self).__init__(is_2d=True, params=params, n_classes=n_classes,
                                             separated_inputs=separated_inputs)

        # Define attributes
        if params is None:
            self.layers = [64, 64, 64]
            self.kernel_size = 3
            self.hidden_dim = 64
            self.lr = 0.001
            self.p_drop = 0.5
            self.n_extra_fc_after_conv = 0
            self.n_extra_fc_final = 0

        layer_dims = [1] + self.layers
        if separated_inputs:
            self.layer_dims = {"g": layer_dims,
                               "h": layer_dims,
                               "f": layer_dims}
        else:
            self.layer_dims = {"a": layer_dims}

        # Layers
        for block in self.blocks:
            for i in range(len(self.layer_dims[block]) - 1):
                self.__dict__["conv_" + block + str(i)] = nn.Conv2d(self.layer_dims[block][i],
                                                                    self.layer_dims[block][i + 1],
                                                                    kernel_size=self.kernel_size, stride=1)
                self.__dict__["relu_" + block + str(i)] = nn.ReLU()
                self.__dict__["pool_" + block + str(i)] = nn.MaxPool2d(kernel_size=(1, 2))
                self.__dict__["drop_" + block + str(i)] = nn.Dropout2d(p=self.p_drop)

            self.__dict__["fc_" + block + "0"] = nn.Linear(self.layer_dims[block][-1], self.hidden_dim)
            self.__dict__["relu_" + block + "0"] = nn.ReLU()
            for i in range(self.n_extra_fc_after_conv):
                self.__dict__["fc_" + block + str(i + 1)] = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.__dict__["relu_" + block + str(i + 1)] = nn.ReLU()

        self.fc0 = nn.Linear(self.out_fc_dim, out_features=self.hidden_dim)
        self.relu0 = nn.ReLU()
        for i in range(self.n_extra_fc_final):
            self.__dict__["fc" + str(i + 1)] = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.__dict__["relu" + str(i + 1)] = nn.ReLU()
