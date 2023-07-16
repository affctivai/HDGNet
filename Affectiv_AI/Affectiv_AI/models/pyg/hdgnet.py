import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
from typing import Union


class Aggregator(nn.Module):
    def __init__(self, region_list, in_channels):
        super(Aggregator, self).__init__()

        self.region_list = region_list
        self.in_channels = in_channels

        self.local_fc = nn.ModuleList()
        # Linear mapping of local features
        for region_index in range(len(self.region_list)):
            self.local_fc.append(
                    nn.Linear(len(self.region_list[region_index]) * self.in_channels, self.in_channels)) # Stacked bands

    def forward(self, x):
        output = []
        for region_index in range(len(self.region_list)):
            region_x = x[:, self.region_list[region_index], :].flatten(start_dim=1)  # get data from a local graph
            aggr_region_x = self.local_fc[region_index](region_x)
            output.append(aggr_region_x)
        return torch.stack(output, dim=1)


class Global_GCN(nn.Module):
    def __init__(self,
                 num_electrodes: int,
                 in_channels: int,
                 out_channels: int,
                 g_fc: int,
                 get_adj:bool):
        super(Global_GCN, self).__init__()

        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.get_adj = get_adj

        self.fc = nn.Linear(self.in_channels, g_fc)

        self.gconv = DenseSAGEConv(self.in_channels, self.out_channels)

    def forward(self, x):
        xa = torch.sigmoid(self.fc(x))
        adj = torch.matmul(xa, xa.transpose(2, 1))  # get the adjacent matrix by multiply the feature

        adj = torch.softmax(adj, 2)  # scale the adjacent matrix to [0,1]

        x = F.relu(self.gconv(x, adj))

        if self.get_adj:
            return x, adj
        else:
            return x

class Local_GCN(nn.Module):
    def __init__(self, region_mask,
                 num_electrodes: int,
                 in_channels: int,
                 out_channels: int,
                 l_fc: int,
                 get_adj:bool):
        super(Local_GCN, self).__init__()
        self.region_mask = region_mask

        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.get_adj = get_adj

        self.fc = nn.Linear(1, l_fc)

        self.gconv = DenseSAGEConv(self.in_channels, self.out_channels)

    def forward(self, x, band_select):
        xa = torch.sigmoid(self.fc(x[:, :, band_select].unsqueeze(2)))
        adj = torch.matmul(xa, xa.transpose(2, 1))  # get the adjacent matrix by multiply the feature
        adj = torch.softmax(adj, 2)  # scale the adjacent matrix to [0,1]

        adj = adj * torch.from_numpy(self.region_mask).unsqueeze(0).to(adj.device).to(torch.float32)

        x = F.relu(self.gconv(x, adj))

        if self.get_adj:
            return x, adj
        else:
            return x


class HDGNet(nn.Module):
    r'''
    Hierarchical Dynamic Local-Global-Graph Representation Learning for EEG Emotion Recognition (HDGNet). For more details, please refer to the following information.

    - Paper: Li H and Kim B. Hierarchical Dynamic Local-Global-Graph Representation Learning for EEG Emotion Recognition.
    - URL: ???
    - Related Project: https://github.com/????

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python
        for SEED dataset
        dataset = SEEDFeatureDataset(io_path=f'./seed',
                                 root_path='./ExtractedFeatures'),
                                 feature=['de_LDS'],
                                 online_transform=transforms.Compose([transforms.ToTensor()]),
                                 label_transform=transforms.Compose([
                                     transforms.Select('emotion'),
                                     transforms.Lambda(lambda x: int(x) + 1)]),
                                 num_worker=1,
                                 io_size=1 * 1024 * 1024 * 1024)

        model = HDGNet(region_mask=SEED_POSTERIOR_REGION_MASK_MATRIX,
                      region_list=SEED_POSTERIOR_REGION_LIST,
                      num_electrodes=62,
                      in_channels=5,
                      hid_channels=5,
                      out_channels=10,
                      num_classes=num_classes).to(device)

        for DREAMER dataset
        dataset = DREAMERDataset(io_path=f'./dreamer',
                             mat_path='./DREAMER.mat',
                             offline_transform=transforms.Compose([
                                 transforms.BandDifferentialEntropy(band_dict={
                                  "theta": [4, 8],
                                  "alpha": [8, 13],
                                  "beta": [13, 20]
                                  }),
                             ]),
                             online_transform=transforms.ToTensor(),
                             label_transform=transforms.Compose([
                                 transforms.Select('arousal'),
                                 transforms.Binary(4.0),
                             ]),
                             num_worker=1,
                             io_size=0.5 * 1024 * 1024 * 1024,
                             io_mode='pickle')

        model = HDGNet(region_mask=DEAP_GENERAL_REGION_MASK_MATRIX,
                        region_list=DEAP_GENERAL_REGION_LIST,
                        num_electrodes=14,
                        in_channels=3,
                        hid_channels=3,
                        out_channels=3,
                        num_classes=num_classes).to(device)

    Args:
            region_mask (torch.Tensor): The mask matrix for localizing, where 1.0 means the node can adjacent and 0.0 means the node can't adjacent. The matrix shape should be [num_electrodes, num_electrodes].
            region_list(list): The local graph structure defined according to the 10-20 system, where the electrodes are divided into different brain regions. (defualt: :obj: )
            num_electrodes (int): The number of electrodes. (defualt: :obj: `62`)
            in_channels (int): The feature dimension of each electrode. (defualt: :obj: `5`)
            hid_channels (int): The number of hidden nodes in the local GNN layer. (defualt: :obj: `5`)
            out_channels (int): The number of hidden nodes in the global GNN layer.(defualt: :obj: `10`)
            num_classes (int): The number of classes to predict.(defualt: :obj: `3`)
            dropout (float): Probability of an element to be zeroed in the dropout layers at the output fully-connected layer. (defualt: :obj: `0.5`)
            l_fc (int): the output dimension of a learned linear layer in constructing local adjacent matrix. (defualt: :obj: `1`)
            g_fc (int): the output dimension of a learned linear layer in constructing global adjacent matrix. (defualt: :obj: `1`)
            get_adj (bool): Whether to return the learned local and global adjacency matrix.
    '''
    def __init__(self,
                 region_mask, region_list,
                 num_electrodes: int = 62,
                 in_channels: int = 5,
                 hid_channels: int = 5,
                 out_channels: int = 10,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 l_fc: int = 1,
                 g_fc: int = 1,
                 get_adj: bool = False):
        super(HDGNet, self).__init__()

        self.region_mask = region_mask
        self.region_list = region_list

        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.l_fc = l_fc
        self.g_fc = g_fc

        self.get_adj = get_adj

        self.lgcn = nn.ModuleList()
        # local graph: graph convolution for different frequency bands based on different local graph
        for i in range(self.in_channels):
            self.lgcn.append(
                Local_GCN(self.region_mask, self.num_electrodes,
                          self.in_channels, self.hid_channels, self.l_fc, self.get_adj))

        # global graph: graph convolution between functional area
        self.aggregate = Aggregator(self.region_list, self.in_channels*self.hid_channels) # the size of output dimension is self.in_channels*self.out_channels
        num_region = len(self.region_list)

        self.bn_g1 = nn.BatchNorm1d(num_region)
        self.bn_g2 = nn.BatchNorm1d(num_region)

        self.ggcn = Global_GCN(num_region, self.in_channels*self.hid_channels,
                               self.out_channels, self.g_fc, self.get_adj)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_region * self.out_channels),
            nn.Dropout(dropout),
            nn.Linear(num_region * self.out_channels, self.num_classes))  # classification


    def forward(self,data: torch.Tensor) -> torch.Tensor:

        if self.get_adj:
            for i in range(len(self.lgcn)):
                one_x, one_local_adj = self.lgcn[i](data, i)
                if i == 0:
                    x = one_x
                    local_adj = one_local_adj
                else:
                    x = torch.cat((x, one_x), dim=2)
                    local_adj = torch.cat((local_adj, one_local_adj), 1)

            x = self.aggregate.forward(x)

            x = self.bn_g1(x)
            x, global_adj = self.ggcn(x)
            x = self.bn_g2(x)

            x = x.flatten(start_dim=1)

            # emotion classification
            x = F.softmax(self.fc(x), dim=1)

            return x, local_adj, global_adj

        else:
            for i in range(len(self.lgcn)):
                one_x = self.lgcn[i](data, i)
                if i == 0:
                    x = one_x
                else:
                    x = torch.cat((x, one_x), dim=2)  # (batch, channel, bands*feature)

            # Aggregation function
            x = self.aggregate.forward(x)

            x = self.bn_g1(x)
            x = self.ggcn(x)
            x = self.bn_g2(x)

            x = x.flatten(start_dim=1) # (batches, num_local, features)

            # emotion classification
            x = F.softmax(self.fc(x), dim=1) # (batches, num_local*features)

            return x