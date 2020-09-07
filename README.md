Dataset: https://figshare.com/articles/ISS_data_in_A_spatiotemporal_organ-wide_gene_expression_and_cell_atlas_of_the_developing_human_heart_/10058048/1

```bash
conda activate
jupyter notebook --ip=127.0.0.1 --no-browser

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

./vips-dev-8.10/bin/vips.exe dzsave ./results/nuclei_heart.tif --tile-size=254 --overlap=1 --depth onepixel --suffix .jpg[Q=90] heart
```

- `preprocess.py`: fig 7, s5
- `graphsage.py` `dgi.py` `gcn.py`: StellarGraph
- `pyg-sage-rw.py` `pyg-sage-dgi.py` `pyg-gcn-rw.py` `pyg-gcn-dgi.py`: PyG
- `cluster.py`: Louvain, UMAP
- `plot-umap.py`: fig 2, 3
- `hm.ipynb`: fig s1, 4, 8, s2
- `paga.py`: fig 5, s3
- `analysis.ipynb`: fig 6, 9, 10, 11
- `pyg-ge.py`: fig s4
- `stitch.py`: TissUUmaps