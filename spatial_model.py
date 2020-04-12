import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_depth, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_depth = hidden_depth
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.th = nn.Tanh()
        self.sg = nn.Sigmoid()
        self.fcns = nn.ModuleList()   #collections.OrderedDict()
        self.bns = nn.ModuleList()   #collections.OrderedDict()
        for i in range(self.hidden_depth):
            #self.fcns.append(nn.Linear(hidden_size, hidden_size).to(device))
            self.fcns.append(nn.Linear(hidden_size, hidden_size))
            #self.bns.append(nn.BatchNorm1d(hidden_size).to(device))
            self.bns.append(nn.BatchNorm1d(hidden_size))
    def forward(self, x):
        out = self.fc1(x)
        for i in range(self.hidden_depth):
            out = self.fcns[i](out)
        #    #out = self.bns[i](out)
            out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sp(out)
        return out


