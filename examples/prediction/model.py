import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
)

from layer import GNN_node, GNN_node_Virtualnode

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x
    
class GNN(torch.nn.Module):

    def __init__(
        self,
        num_task,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0.5,
        JK="last",
        graph_pooling="sum",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_task = num_task
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNN_node(
                num_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                )
            )
        else:
            raise ValueError("Invalid graph pooling type.")

        self.predictor = MLP(
            emb_dim + 2048, hidden_features=4 * emb_dim, out_features=num_task
        )

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        h_graph = torch.cat([h_graph, batched_data.fp.type_as(h_graph)], dim=1)
        return self.predictor(h_graph)

if __name__ == "__main__":
    pass
