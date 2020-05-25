# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from textwrap import wrap


# %%
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# %%
data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(result_dir, exist_ok=True)
fig_dir = os.path.join(os.getcwd(), 'figures')
os.makedirs(fig_dir, exist_ok=True)


# %%
df = {}
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    df[name] = pd.read_csv(os.path.join(data_dir, 'spots_PCW{}.csv'.format(name)))
    print(df[name].shape)
    df[name]['pcw'] = int(name[0])
    df[name]['section'] = int(name[-1])
    df[name] = df[name].set_index('{}_'.format(name) + df[name].index.astype(str))
df_heart = pd.concat(df.values())
print(df_heart['pcw'].value_counts())


# %%
fig, axes = plt.subplots(3, 3, figsize=(21, 21))
name_list = ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '', '9.5_1', '9.5_2', '9.5_3']
for i in range(3):
    for j in range(3):
        name = name_list[3*i + j]
        if name == '':
            fig.delaxes(axes[i,j])
            continue
        subset = df_heart.index.str.startswith(name)
        spot_x = df_heart.loc[subset,'spotX']
        spot_y = df_heart.loc[subset,'spotY']
        spot_y = spot_y.max() - spot_y
        axes[i,j].scatter(
            spot_x, 
            spot_y, 
            s=1, 
            marker='.', 
            linewidths=0
        )
        axes[i,j].set_title(name, fontsize=20)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'heart.png'))
plt.close()


# %%
df_cell = {}
for name in ['6.5_1', '6.5_2']:
    df_cell_segmentation = pd.read_csv(os.path.join(data_dir, 'spots_w_cell_segmentation_PCW{}.csv'.format(name)))
    df_cell_calling = pd.read_csv(os.path.join(data_dir, 'cell_calling_PCW{}.csv'.format(name)))
    df_cell[name] = pd.merge(
        df_cell_segmentation, df_cell_calling[['cell', 'celltype']], 
        how='left', left_on='parent_id', right_on='cell'
    )
    df_cell[name]['pcw'] = int(name[0])
    df_cell[name]['section'] = int(name[-1])
    df_cell[name] = df_cell[name].set_index('{}_'.format(name) + df_cell[name].index.astype(str))
df_heart_cell = pd.concat(df_cell.values())
cell_types = df_heart_cell['celltype'].dropna().unique()
cell_type_id = [(cell_type, int(re.search(r'\((\d+)\)', cell_type).group(1))) for cell_type in cell_types]
cell_type_id.append(('Uncalled', -1))
cell_type_id.sort(key=lambda x: x[1])
cell_type_id = dict(cell_type_id)
df_heart_cell['celltype'].fillna(value='Uncalled', inplace=True)
df_heart_cell['cell_type_id'] = df_heart_cell['celltype'].map(cell_type_id)
df_heart_cell['cell_type_id'].to_csv(os.path.join(result_dir, 'celltype.csv'))
print(df_heart_cell['celltype'].value_counts())
id_cell_type = {v: k for (k, v) in cell_type_id.items()}
id_cell_type[2] = '(2) Fibroblast-like (related to cardiac skeleton connective tissue)'
id_cell_type[3] = '(3) Epicardium-derived cells'
id_cell_type[4] = '(4) Fibroblast-like (smaller vascular development)'
id_cell_type[8] = '(8) Fibroblast-like (larger vascular development)'
id_cell_type[14] = '(14) Cardiac neural crest cells & Schwann progenitor cells'


# %%
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
ct_cmap = [
    '#F8766D', '#E58700', '#C99800', '#A3A500', '#6BB100',
    '#00BA38', '#00BF7D', '#00C0AF', '#00BCD8', '#00B0F6', 
    '#619CFF', '#B983FF', '#E76BF3', '#FD61D1', '#FF67A4'
]
cmap = ListedColormap(ct_cmap)
labels = ['\n'.join(wrap(line, 30)) for line in list(id_cell_type.values())[1:]]
for s in range(2):
    subset = (df_heart_cell['section'] == s+1) & (df_heart_cell['cell_type_id'] >= 0)
    spot_x = df_heart_cell.loc[subset,'spotX']
    spot_y = df_heart_cell.loc[subset,'spotY']
    spot_y = spot_y.max() - spot_y
    scatter = axes[s].scatter(
        spot_x, 
        spot_y, 
        s=1, 
        c=df_heart_cell.loc[subset,'cell_type_id'], 
        marker='.', 
        cmap=cmap, 
        vmin=0, 
        vmax=14, 
        alpha=1.0, 
        linewidths=0
    )
    axes[s].set_title('6.5_' + str(s+1), fontsize=20)
    axes[s].set_xticks([])
    axes[s].set_yticks([])
axes[2].legend(
    handles=scatter.legend_elements(num=None)[0], 
    labels=labels, 
    loc='center left', 
    bbox_to_anchor=(0, 0.5), 
    fontsize=15,
    markerscale=2
)
axes[2].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'cell_type.png'), dpi=200)
plt.close()


# %%
fig, axes = plt.subplots(3, 4, figsize=(28, 21))
ct_cmap = [
    '#F8766D', '#E58700', '#C99800', '#A3A500', '#6BB100',
    '#00BA38', '#00BF7D', '#00C0AF', '#00BCD8', '#00B0F6', 
    '#619CFF', '#B983FF', '#E76BF3', '#FD61D1', '#FF67A4'
]
for ax, cell_type in zip(axes.ravel(), list(id_cell_type.keys())[1:]):
    subset = (df_heart_cell['section'] == 2)
    spot_x = df_heart_cell.loc[subset,'spotX']
    spot_y = df_heart_cell.loc[subset,'spotY']
    spot_y = spot_y.max() - spot_y
    ax.scatter(
        spot_x[df_heart_cell['cell_type_id'] != cell_type], 
        spot_y[df_heart_cell['cell_type_id'] != cell_type], 
        s=1, 
        c='gray', 
        marker='.', 
        alpha=0.1, 
        linewidths=0
    )
    ax.scatter(
        spot_x[df_heart_cell['cell_type_id'] == cell_type], 
        spot_y[df_heart_cell['cell_type_id'] == cell_type], 
        s=1, 
        c=ct_cmap[cell_type], 
        marker='.', 
        alpha=1.0, 
        linewidths=0
    )
    ax.set_title('\n'.join(wrap(id_cell_type[cell_type], 30)), fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, 'cell_type_12.png'), dpi=100)
plt.close()


# %%
df_edge = {}
pct = 99
distances_all = []
print(pct)
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    spots = df[name][['spotX', 'spotY']]
    kdtree = KDTree(spots)
    distances, _ =  kdtree.query(spots, k=2)
    distances_all.append(distances)
distances_all = np.concatenate(distances_all, axis=0)
d_max = np.percentile(distances_all[:,1], pct)
print(d_max)
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    spots = df[name][['spotX', 'spotY']]
    kdtree = KDTree(spots)
    # distances, _ =  kdtree.query(spots, k=2)
    # d_max = np.percentile(distances[:,1], pct)
    # print(d_max)
    ind, dist = kdtree.query_radius(spots, d_max, return_distance=True)
    df_edge[name] = pd.DataFrame(
        data=[
            (spots.index[i], spots.index[j], d)
            for i in range(len(spots))
            for (j, d) in zip(ind[i], dist[i]) if i < j], 
        columns=['source', 'target', 'distance']
    )
    # bandwidth = df_edge[name]['distance'].mean()
    # df_edge[name]['weight'] = np.exp(
    #     (-np.square(df_edge[name]['distance'])) 
    #     / (2*np.square(bandwidth)))
    # df_edge[name]['weight'] = 1 / df_edge[name]['distance']
    # df_edge[name] = df_edge[name].drop('distance', axis=1)
    df_edge[name] = df_edge[name].set_index('{}_'.format(name) + df_edge[name].index.astype(str))
df_edges = pd.concat(df_edge.values())
df_edges = df_edges.reset_index()


# %%
g = nx.from_pandas_edgelist(df_edges, edge_attr='index')
n_cc = 6
print(n_cc)
for cc in nx.connected_components(g.copy()):
    if len(cc) < n_cc:
        g.remove_nodes_from(cc)
print(g.number_of_nodes())
print(g.number_of_edges() / g.number_of_nodes() * 2)


# %%
df_nodes = df_heart.loc[list(g.nodes), 'gene']
df_nodes.to_csv(os.path.join(result_dir, 'nodes.csv'))
df_edges = nx.to_pandas_edgelist(g)
df_edges = df_edges.set_index('index')
df_edges.to_csv(os.path.join(result_dir, 'edges.csv'))


# %%
df_nodes = pd.read_csv(os.path.join(result_dir, 'nodes.csv'), index_col=0)
df_celltype = pd.read_csv(os.path.join(result_dir, 'celltype.csv'), index_col=0)
df_taglist = pd.read_csv(os.path.join(data_dir, 'taglist_heart.csv'), names=['tag', 'gene'])
enc = OneHotEncoder(sparse=False).fit(df_taglist['gene'].to_numpy().reshape(-1, 1))
df_nodes = df_nodes.loc[df_nodes.index.str.startswith('6')]
df_nodes = pd.DataFrame(data=enc.transform(df_nodes['gene'].to_numpy().reshape(-1, 1)), index=df_nodes.index)
df_nodes = pd.concat([df_nodes, df_celltype], axis=1).reindex(df_nodes.index)
X = df_nodes.loc[df_nodes['cell_type_id'] >= 0].to_numpy()
print(X.shape)
X, y = X[:,:-1], X[:,-1]
clf = LogisticRegression(verbose=False, n_jobs=-1)
print(cross_val_score(clf, X, y, scoring='f1_weighted', cv=5))


