"""GNNExplainer by PyTorch Geometric."""

# %%
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
from pathlib import Path
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv, GNNExplainer
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from torch_cluster import random_walk
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from scipy.cluster.vq import vq

# %% [markdown]
# Load graph data and clustering result.

# %%
data_dir = Path.cwd() / "data"
df_taglist = pd.read_csv(data_dir / "taglist_heart.csv", names=["tag", "gene"])
enc = OneHotEncoder(sparse=False).fit(df_taglist["gene"].to_numpy().reshape(-1, 1))

result_dir = Path.cwd() / "results"
df_nodes = pd.read_csv(result_dir / "nodes.csv", index_col=0)
df_nodes = pd.DataFrame(
    data=enc.transform(df_nodes["gene"].to_numpy().reshape(-1, 1)), index=df_nodes.index
)
df_edges = pd.read_csv(result_dir / "edges.csv", index_col=0)
adata_name = "SAGE-20201013.h5ad"
adata = sc.read(result_dir / adata_name)

# df_nodes = df_nodes[df_nodes.index.str.startswith('6')]
# df_edges = df_edges[df_edges.index.str.startswith('6')]

index_dict = dict(zip(df_nodes.index, range(len(df_nodes))))
df_edges_index = df_edges[["source", "target"]].applymap(index_dict.get)
x = torch.tensor(df_nodes.to_numpy(), dtype=torch.float)
edge_index = torch.tensor(df_edges_index.to_numpy(), dtype=torch.long)
edge_index = to_undirected(edge_index.t().contiguous())
y = torch.tensor(adata.obs["spage2vec"].to_numpy(), dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)
data.adj_t = SparseTensor(
    row=edge_index[0], col=edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes)
).t()
print(data)
print(data.num_edges / data.num_nodes)

num_clusters = adata.obs["spage2vec"].max() + 1
centroids = np.array(
    [adata.X[adata.obs["spage2vec"] == i].mean(axis=0) for i in range(num_clusters)]
)
closest_idx, _ = vq(centroids, adata.X)

# %% [markdown]
# Train the model in a supervised way.

# %%
class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.num_layers = 2
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)


hidden_channels = 32
epochs = 1000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
model = SAGE(data.num_features, hidden_channels, num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(1, epochs):
    model.train()
    optimizer.zero_grad()
    log_logits = model(data.x, data.adj_t)
    loss = F.nll_loss(log_logits, data.y, ignore_index=-1)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: {:03d}, Loss: {:.4f}".format(epoch, loss))


model.eval()
log_logits = model(data.x, data.adj_t)
y_pred = log_logits.max(1)[1].cpu()
y_true = data.y.cpu()
score = f1_score(y_true, y_pred, average="weighted")
print("Score: {:.4f}".format(score))

# %% [markdown]
# Make plots.

# %%
fig_dir = Path.cwd() / "figures"
fig_dir.mkdir(exist_ok=True)
df_feat = pd.DataFrame(
    data=np.zeros((data.num_features, num_clusters)), index=df_taglist["gene"]
)
explainer = GNNExplainer(model, epochs=100, lr=1e-2, log=True)
for i in range(num_clusters):
    node_idx = int(closest_idx[i])
    node_feat_mask, edge_mask = explainer.explain_node(
        node_idx, data.x, data.edge_index
    )
    df_feat.iloc[:, i] = node_feat_mask.cpu().numpy()
    # ax, G = explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
    # plt.savefig(fig_dir / 'explainer-{}.png'.format(i))
fig, ax = plt.subplots(figsize=(10, 10))
g = sns.clustermap(
    df_feat.sort_index(),
    method="average",
    metric="correlation",
    row_cluster=False,
    cmap="viridis",
    xticklabels=True,
    yticklabels=True,
)
g.savefig(fig_dir / "explainer-sup.png")
