from torch import nn
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self, n_layers, out_features, dropout_p):
        super(Net, self).__init__()
        in_features = 28 * 28
        CLASSES = 10
        layers = []
        for i in range(n_layers):
#            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))

            in_features = out_features
        layers.append(nn.Linear(in_features, CLASSES))
        layers.append(nn.LogSoftmax(dim=1))
        
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)
