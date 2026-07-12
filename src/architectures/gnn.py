import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, Set2Set, global_add_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from src.config.features import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

POOLING_CHOICES = ("mean", "sum", "set2set", "attention")


class SolubilityGNN(nn.Module):
    """GINE message-passing network for graph-level solubility regression.

    Sized to ESOL (~1,100 molecules) per the EDA: small hidden dim, few
    layers, dropout in the head -- deeper/wider nets overfit and
    over-smooth on graphs this small.
    """

    def __init__(self, hidden=128, n_layers=3, dropout=0.2, pooling="mean"):
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"pooling must be one of {POOLING_CHOICES}, got {pooling!r}")
        self.pooling = pooling

        self.node_embed = nn.Linear(NODE_FEATURE_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_FEATURE_DIM, hidden)  # GINE needs edge dim = node dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        if pooling == "mean":
            self.pool = global_mean_pool
            pooled_dim = hidden
        elif pooling == "sum":
            self.pool = global_add_pool
            pooled_dim = hidden
        elif pooling == "set2set":
            self.pool = Set2Set(hidden, processing_steps=3)
            pooled_dim = 2 * hidden  # Set2Set concatenates query + read vectors
        else:  # attention
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden, 1))
            pooled_dim = hidden

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x = self.node_embed(data.x)
        e = self.edge_embed(data.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, e)))
        x = self.pool(x, data.batch)
        return self.head(x).squeeze(-1)


class MultiTaskToxGNN(nn.Module):
    """GINE message-passing network for multi-task binary toxicity
    classification (Tox21's 12 assays).

    Same backbone as SolubilityGNN -- the EDA's assay-assay correlation
    (notebooks/tox21_eda.ipynb, section 4) found real cross-assay signal,
    which is what justifies one shared trunk feeding N_TASKS output heads
    instead of N_TASKS independent models.
    """

    def __init__(self, n_tasks, hidden=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.node_embed = nn.Linear(NODE_FEATURE_DIM, hidden)
        self.edge_embed = nn.Linear(EDGE_FEATURE_DIM, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, data):
        x = self.node_embed(data.x)
        e = self.edge_embed(data.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, e)))
        x = global_mean_pool(x, data.batch)
        return self.head(x)  # [batch, n_tasks] raw logits -- no sigmoid, handled in the loss
