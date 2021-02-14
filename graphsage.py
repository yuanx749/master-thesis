"""GraphSAGE by StellarGraph."""

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
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer.graphsage import (
    MeanAggregator,
    MeanPoolingAggregator,
    MaxPoolingAggregator,
    AttentionalAggregator,
)
from stellargraph.layer.link_inference import LinkEmbedding
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
df_edges = df_edges[["source", "target"]]


# %% [markdown]
# Create StellarGraph.

# %%
# df_nodes = df_nodes[df_nodes.index.str.startswith('6')]
# df_edges = df_edges[df_edges.index.str.startswith('6')]
g = StellarGraph(nodes=df_nodes, edges=df_edges)
print(g.info())
print(len(g.edges()) / len(g.nodes()) * 2)


# %% [markdown]
# Create the UnsupervisedSampler.

# %%
number_of_walks = 1
length = 2
unsupervised_samples = UnsupervisedSampler(
    g, nodes=list(g.nodes()), length=length, number_of_walks=number_of_walks
)


# %% [markdown]
# Create a node pair generator.

# %%
batch_size = 64
num_samples = [10, 5]
generator = GraphSAGELinkGenerator(g, batch_size, num_samples, weighted=True)
train_gen = generator.flow(unsupervised_samples)


# %% [markdown]
# Build the model.

# %%
layer_sizes = [32, 32]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes,
    generator=generator,
    aggregator=AttentionalAggregator,
    bias=True,
    dropout=0.0,
    normalize="l2",
    # kernel_regularizer='l1'
)
x_inp, x_out = graphsage.in_out_tensors()
prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)


# %%
model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)
model.summary()


# %% [markdown]
# Train the model.

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
    train_gen,
    epochs=100,
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
model_name = os.path.join(model_dir, time.strftime("sage-%Y%m%d.h5"))
model.save(model_name)


# %%
model_dir = os.path.join(os.getcwd(), "models")
model_name = os.path.join(model_dir, time.strftime("sage-%Y%m%d.h5"))
custom_objects = {
    "MeanAggregator": MeanAggregator,
    "AttentionalAggregator": AttentionalAggregator,
    "LinkEmbedding": LinkEmbedding,
}
model = keras.models.load_model(model_name, custom_objects=custom_objects)
file_ = h5py.File(model_name, mode="r")
print(file_.attrs.get("model_config"))
print(file_.attrs.get("training_config"))
file_.close()


# %% [markdown]
# Save node embedding.

# %%
x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
model_name = os.path.join(model_dir, time.strftime("sage-embedding-%Y%m%d.h5"))
embedding_model.save(model_name)


# %%
custom_objects = {
    "MeanAggregator": MeanAggregator,
    "AttentionalAggregator": AttentionalAggregator,
}
embedding_model = keras.models.load_model(model_name, custom_objects=custom_objects)
# embedding_model.summary()


# %%
node_gen = GraphSAGENodeGenerator(g, batch_size, num_samples, weighted=True).flow(
    g.nodes()
)


# %%
node_embeddings = embedding_model.predict(node_gen, verbose=2)


# %%
result_dir = os.path.join(os.getcwd(), "results")
os.makedirs(result_dir, exist_ok=True)
embedding_name = time.strftime("sage-embedding-%Y%m%d-%H%M%S.npy")
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
