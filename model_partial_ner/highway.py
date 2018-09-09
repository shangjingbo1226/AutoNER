"""
.. module:: highway
    :synopsis: highway layers 
    
.. moduleauthor:: Liyuan Liu, Jingbo Shang
"""
import torch
import torch.nn as nn
import model_partial_ner.utils as utils

class highway(nn.Module):
    """
    Highway layers

    Parameters
    ----------
    size: ``int``, required.
        Input and output dimension.
    num_layers: ``int``, required.
        Number of layers.
    droprate: ``float``, required.
        Dropout ratio
    """  
    def __init__(self, size, num_layers = 1, droprate = 0.5):
        super(highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=droprate)

        for i in range(num_layers):
            tmptrans = nn.Linear(size, size)
            tmpgate = nn.Linear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def rand_ini(self):
        """
        random initialization
        """
        for i in range(self.num_layers):
            utils.init_linear(self.trans[i])
            utils.init_linear(self.gate[i])

    def forward(self, x):
        """
        update statics for f1 score

        Parameters
        ----------
            x (ins_num, hidden_dim): input tensor
        Returns
        ----------
        output: ``torch.FloatTensor``.
            output tensor (ins_num, hidden_dim)
        """
        
        
        g = torch.sigmoid(self.gate[0](x))
        h = nn.functional.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = self.dropout(x)
            g = torch.sigmoid(self.gate[i](x))
            h = nn.functional.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x