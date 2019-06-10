# Notice

### For the future updates of ABRW method, please see https://github.com/houchengbin/OpenANE
<br>

# ABRW: Attributed Biased Random Walks
ABRW is an Attributed Network/Graph Embedding (ANE) method, which takes network structural information and node attribute information as the input, and generates unified low-dim node embeddings for each node in the network as the output. The node embeddings can be then fed into various different downstream tasks e.g. node classification and link prediction. The off-the-shelf Machine Learning techniques and distance/similarity metrics can be easily applied in the downstream tasks, since the resulting node embeddings are just some isolated low-dim data points in Euclidean space.

by Chengbin HOU 2018 chengbin.hou10(AT)foxmail.com

For more details, please have a look at our paper https://arxiv.org/abs/1811.11728. And if you find ABRW or this framework is useful for your research, please consider citing it.
```
@article{hou2019abrw,
  title={Attributed Network Embedding for Incomplete Attributed Networks},
  author={Hou, Chengbin and He, Shan and Tang, Ke},
  journal={arXiv preprint arXiv:1811.11728v2},
  year={2019}
}
```

# Usages
### Install necessary packages
```bash
pip install -r requirements.txt
```
Note: python 3.6 or above is required due to the [new print() feature](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings)
### Run ABRW method with default parameters
```bash
python src/main.py --method abrw
```
### Try other methods
abrw; aane; tadw; attrpure; attrcomb; deepwalk; node2vec
### Change parameters in each method
Please see main.py

## Datasets
### Default
Cora (a citation network)
### Your own dataset?
#### FILE for structural information (each row):
adjlist: node_id1 node_id2 node_id3 -> (the edges between (id1, id2) and (id1, id3)) 

OR edgelist: node_id1 node_id2 weight(optional) -> one edge (id1, id2)
#### FILE for attribute information (each row):
node_id1 attr1 attr2 ... attrM

#### FILE for label (each row):
node_id1 label(s)

## Acknowledgement
Thanks to Zeyu DONG for helpful discussions. And thanks to [the excellent project](https://github.com/thunlp/OpenNE), so that we can have a good starting point to carry on our project.
