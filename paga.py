"""Create PAGA graphs on the merged clusters."""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns


# %%
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# %%
sc.logging.print_header()
sc.settings.verbosity = 4
sc.set_figure_params(dpi_save=200)


# %%
data_dir = Path.cwd() / "data"
result_dir = Path.cwd() / "results"


# %%
model = "SAGE"
adata_name = "{}-20201013.h5ad".format(model)
adata = sc.read(result_dir / adata_name)
print(adata_name)
print(adata)


# %%
adata.obs["spage2vec"] = adata.obs["spage2vec"].astype("category")
adata_s = adata[adata.obs["spage2vec"].astype("int") > -1]
sc.tl.paga(adata_s, groups="spage2vec")
sc.pl.paga(
    adata_s,
    threshold=0.2,
    fontsize=7,
    node_size_scale=0.5,
    node_size_power=0.5,
    edge_width_scale=0.5,
    save=".png",
)
print(adata_s)
sc.tl.umap(adata_s, min_dist=0.5, init_pos="paga", random_state=42)
sc.pl.paga_compare(adata_s, threshold=0.2, save=".png")
