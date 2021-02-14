"""Deep Graph Infomax GraphSAGE by StellarGraph."""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# %%
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py

from stellargraph import StellarGraph
from stellargraph.mapper import CorruptedGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, DeepGraphInfomax
from stellargraph.layer.graphsage import (
    MeanAggregator,
    MeanPoolingAggregator,
    MaxPoolingAggregator,
    AttentionalAggregator,
)
from stellargraph.layer.deep_graph_infomax import DGIReadout, DGIDiscriminator
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder


# %% [markdown]
# Load data.

# %%
data_dir = os.path.join(os.getcwd(), "data")
df_taglist = pd.read_csv(
    os.path.join(data_dir, "taglist_heart.csv"), names=["tag", "gene"]
)
enc = OneHotEncoder(sparse=False).fit(df_taglist["gene"].to_numpy().reshape(-1, 1))


# %%
result_dir = os.path.join(os.getcwd(), "results")
df_nodes = pd.read_csv(os.path.join(result_dir, "nodes.csv"), index_col=0)
df_nodes = pd.DataFrame(
    data=enc.transform(df_nodes["gene"].to_numpy().reshape(-1, 1)), index=df_nodes.index
)
df_edges = pd.read_csv(os.path.join(result_dir, "edges.csv"), index_col=0)


# %% [markdown]
# Create StellarGraph.

# %%
# df_nodes = df_nodes[df_nodes.index.str.startswith('6')]
# df_edges = df_edges[df_edges.index.str.startswith('6')]
g = StellarGraph(nodes=df_nodes, edges=df_edges[["source", "target"]])
print(g.info())
print(len(g.edges()) / len(g.nodes()) * 2)


# %% [markdown]
# Data generater.

# %%
batch_size = 64
num_samples = [20, 10]
layer_sizes = [32, 32]
graphsage_generator = GraphSAGENodeGenerator(g, batch_size, num_samples)
graphsage_model = GraphSAGE(
    layer_sizes=layer_sizes,
    generator=graphsage_generator,
    aggregator=AttentionalAggregator,
    bias=True,
    dropout=0.0,
    normalize="l2",
    activations=["relu", "linear"],
    # kernel_regularizer='l1'
)
corrupted_generator = CorruptedGenerator(graphsage_generator)
gen = corrupted_generator.flow(g.nodes())


# %% [markdown]
# Model creation and training.

# %%
infomax = DeepGraphInfomax(graphsage_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()
model = keras.Model(inputs=x_in, outputs=x_out)
model.compile(
    loss=tf.nn.sigmoid_cross_entropy_with_logits,
    optimizer=keras.optimizers.Adam(lr=1e-3),
)
model.summary()


# %%
es_callback = keras.callbacks.EarlyStopping(
    monitor="loss", min_delta=0, patience=1, restore_best_weights=True
)


# %%
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S.csv"))
)


# %%
history = model.fit(
    gen,
    epochs=30,
    verbose=2,
    use_multiprocessing=False,
    workers=1,
    shuffle=True,
    callbacks=[es_callback, csv_logger],
)


# %% [markdown]
# Save the model.

# %%
model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(model_dir, exist_ok=True)
model_name = os.path.join(model_dir, time.strftime("dgi-%Y%m%d.h5"))
model.save(model_name)


# %%
model_dir = os.path.join(os.getcwd(), "models")
model_name = os.path.join(model_dir, time.strftime("dgi-%Y%m%d.h5"))
custom_objects = {
    "MeanAggregator": MeanAggregator,
    "MeanPoolingAggregator": MeanPoolingAggregator,
    "MaxPoolingAggregator": MaxPoolingAggregator,
    "AttentionalAggregator": AttentionalAggregator,
    "DGIReadout": DGIReadout,
    "DGIDiscriminator": DGIDiscriminator,
    "sigmoid_cross_entropy_with_logits_v2": tf.nn.sigmoid_cross_entropy_with_logits,
}
model = keras.models.load_model(model_name, custom_objects=custom_objects)
file_ = h5py.File(model_name, mode="r")
print(file_.attrs.get("model_config"))
print(file_.attrs.get("training_config"))
file_.close()


# %% [markdown]
# Save node embedding.

# %%
x_emb_in, x_emb_out = graphsage_model.in_out_tensors()
embedding_model = keras.Model(inputs=x_emb_in, outputs=x_emb_out)
model_name = os.path.join(model_dir, time.strftime("dgi-embedding-%Y%m%d.h5"))
embedding_model.save(model_name)


# %%
custom_objects = {
    "MeanAggregator": MeanAggregator,
    "MeanPoolingAggregator": MeanPoolingAggregator,
    "MaxPoolingAggregator": MaxPoolingAggregator,
    "AttentionalAggregator": AttentionalAggregator,
}
embedding_model = keras.models.load_model(model_name, custom_objects=custom_objects)
# embedding_model.summary()


# %%
node_embeddings = embedding_model.predict(
    graphsage_generator.flow(g.nodes()), verbose=2
)


# %%
result_dir = os.path.join(os.getcwd(), "results")
os.makedirs(result_dir, exist_ok=True)
embedding_name = time.strftime("dgi-embedding-%Y%m%d-%H%M%S.npy")
np.save(os.path.join(result_dir, embedding_name), node_embeddings)
print(embedding_name)


# %%
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df_celltype = pd.read_csv(os.path.join(result_dir, "celltype.csv"), index_col=0)
df_nodes = pd.DataFrame(data=node_embeddings, index=df_nodes.index)
df_nodes = pd.concat([df_nodes, df_celltype], axis=1).reindex(df_nodes.index)
X = df_nodes.loc[df_nodes["cell_type_id"] >= 0].to_numpy()
print(X.shape)
X, y = X[:, :-1], X[:, -1]
clf = LogisticRegression(verbose=False, n_jobs=-1)
# clf = LinearSVC()
print(cross_val_score(clf, X, y, scoring="f1_weighted", cv=5))
