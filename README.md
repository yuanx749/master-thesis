# master-thesis
This repository contains the code for my [master thesis](http://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1508866).

## Data
Download the original dataset [here](https://figshare.com/articles/ISS_data_in_A_spatiotemporal_organ-wide_gene_expression_and_cell_atlas_of_the_developing_human_heart_/10058048/1).

## Set up environment
Install Miniconda. `conda env create -f environment.yml`. Or install dependencies as below.
```bash
conda install -c conda-forge notebook
conda install -c stellargraph stellargraph
pip install pydot pydotplus graphviz

pip install scanpy[louvain,leiden]
pip install pynndescent
pip install phate

conda install -c conda-forge ipywidgets
conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install torch-sparse==0.6.6 -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install torch-cluster==1.5.6 -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install torch-geometric==1.6.0
```

Some useful commands:
```bash
conda activate
jupyter notebook --ip=127.0.0.1 --no-browser

./vips-dev-8.10/bin/vips.exe dzsave ./results/nuclei_heart.tif --tile-size=254 --overlap=1 --depth onepixel --suffix .jpg[Q=90] heart
```

## Scripts
All the scripts are self-contained and can run in **notebook** mode. The function of each script and the corresponding figures produced in the thesis are listed below. Refer to the scripts for more descriptions.
- `preprocess.py`: fig 7, s5
- `graphsage.py`, `dgi.py`, `gcn.py`: StellarGraph (not used in the thesis)
- `pyg-sage-rw.py`, `pyg-sage-dgi.py`, `pyg-gcn-rw.py`, `pyg-gcn-dgi.py`: **PyG**
- `cluster.py`: Louvain, PAGA, UMAP
- `plot-umap.py`: fig 2, 3
- `hm.ipynb`: fig s1, 4, 8, s2
- `paga.py`: fig 5, s3
- `analysis.ipynb`: fig 6, 9, 10, 11
- `pyg-ge.py`: fig s4
- `stitch.py`: for TissUUmaps
- `run.sh`: simple logging
