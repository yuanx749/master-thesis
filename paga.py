# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import random
import time
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns


# %%
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# %%
sc.logging.print_header()
sc.settings.verbosity = 4
sc.set_figure_params(dpi_save=200)


# %%
data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')


# %%
model = 'SAGE'
adata_name = '{}-20201013.h5ad'.format(model)
adata = sc.read(os.path.join(result_dir, adata_name))
print(adata_name)
print(adata)


# %%
adata.obs['spage2vec'] = adata.obs['spage2vec'].astype('category')
adata_s = adata[adata.obs['spage2vec'].astype('int') > -1]
sc.tl.paga(adata_s, groups='spage2vec')
sc.pl.paga(
    adata_s, 
    threshold=0.2, 
    fontsize=7, 
    node_size_scale=0.5, 
    node_size_power=0.5, 
    edge_width_scale=0.5, 
    save='.png')
print(adata_s)
sc.tl.umap(adata_s, min_dist=0.5, init_pos='paga', random_state=42)
sc.pl.paga_compare(adata_s, threshold=0.2, save='.png')
# sc.pl.umap(
#     adata_s, 
#     color='spage2vec', 
#     legend_loc='on data', 
#     legend_fontsize='xx-small', 
#     title='', 
#     save='.png')
# adata_s.write(os.path.join(result_dir, 'spage2vec.h5ad'))


# adata_s.obs['pcw'] = adata_s.obs['pcw'].astype('category')
# sc.pl.umap(adata_s, color=['pcw'], palette='viridis', save='-pcw.png')
# adata_s.write(os.path.join(result_dir, 'spage2vec.h5ad'))

