'''
STEP1: load data -->
STEP2: prepare data -->
STEP3: learn node embeddings -->
STEP4: downstream evaluations

python src/main.py --method abrw

by Chengbin HOU 2018 <chengbin.hou10@foxmail.com>
'''

import time
import random
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from libnrl.classify import ncClassifier, lpClassifier, read_node_label
from libnrl.graph import *
from libnrl.utils import *
from libnrl import abrw #ANE method; Attributed Biased Random Walk
from libnrl import tadw #ANE method
from libnrl import aane #ANE method
from libnrl import attrcomb #ANE method
from libnrl import attrpure #NE method simply use svd or pca for dim reduction
from libnrl import node2vec #PNE method; including deepwalk and node2vec

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    #-----------------------------------------------general settings--------------------------------------------------
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='graph/network format')
    parser.add_argument('--graph-file', default='data/cora/cora_adjlist.txt',
                        help='graph/network file')
    parser.add_argument('--attribute-file', default='data/cora/cora_attr.txt',
                        help='node attribute/feature file')
    parser.add_argument('--label-file', default='data/cora/cora_label.txt',
                        help='node label file')     
    parser.add_argument('--dim', default=128, type=int,
                        help='node embeddings dimensions')
    parser.add_argument('--task', default='lp_and_nc', choices=['none', 'lp', 'nc', 'lp_and_nc'],
                        help='choices of downstream tasks: none, lp, nc, lp_and_nc')
    parser.add_argument('--link-remove', default=0.1, type=float, 
                        help='simulate randomly missing links if necessary; a ratio ranging [0.0, 1.0]')
    parser.add_argument('--label-reserved', default=0.7, type=float,
                        help='for nc task, train/test split, a ratio ranging [0.0, 1.0]')
    parser.add_argument('--directed', default=False, action='store_true',
                        help='directed or undirected graph')
    parser.add_argument('--weighted', default=False, action='store_true',
                        help='weighted or unweighted graph')
    parser.add_argument('--save-emb', default=False, action='store_true',
                        help='save emb to disk if True')
    parser.add_argument('--emb-file', default='emb/unnamed_node_embs.txt',
                        help='node embeddings file; suggest: data_method_dim_embs.txt')
    #-------------------------------------------------method settings-----------------------------------------------------------
    parser.add_argument('--method', default='abrw', choices=['deepwalk', 'node2vec', 'abrw', 'attrpure', 'attrcomb', 'tadw', 'aane'],
                        help='choices of Network Embedding methods')
    parser.add_argument('--ABRW-topk', default=30, type=int,
                        help='select the most attr similar top k nodes of a node; ranging [0, # of nodes]') 
    parser.add_argument('--ABRW-alpha', default=0.8, type=float,
                        help='balance struc and attr info; ranging [0, 1]')       
    parser.add_argument('--AANE-lamb', default=0.05, type=float,
                        help='balance struc and attr info; ranging [0, inf]')
    parser.add_argument('--AANE-rho', default=5, type=float,
                        help='penalty parameter; ranging [0, inf]')
    parser.add_argument('--AANE-maxiter', default=10, type=int,
                        help='max iter')
    parser.add_argument('--TADW-lamb', default=0.2, type=float,
                        help='balance struc and attr info; ranging [0, inf]') 
    parser.add_argument('--TADW-maxiter', default=10, type=int,
                        help='max iter') 
    parser.add_argument('--AttrComb-mode', default='concat', type=str,
                        help='choices of mode: concat, elementwise-mean, elementwise-max')
    parser.add_argument('--Node2Vec-p', default=0.5, type=float,  #if p=q=1.0 node2vec = deepwalk
                        help='trade-off BFS and DFS; rid search [0.25; 0.50; 1; 2; 4]')             
    parser.add_argument('--Node2Vec-q', default=0.5, type=float,
                        help='trade-off BFS and DFS; rid search [0.25; 0.50; 1; 2; 4]')
    #for walk based methods; some Word2Vec SkipGram parameters are not specified here
    parser.add_argument('--number-walks', default=10, type=int,
                        help='# of random walks of each node')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='length of each random walk')
    parser.add_argument('--window-size', default=10, type=int, 
                        help='window size of skipgram model')
    parser.add_argument('--workers', default=24, type=int,
                        help='# of parallel processes.')
    args = parser.parse_args()
    return args


def main(args):
    g = Graph() #see graph.py for commonly-used APIs and use g.G to access NetworkX APIs
    print(f'Summary of all settings: {args}')


    #---------------------------------------STEP1: load data-----------------------------------------------------
    print('\nSTEP1: start loading data......')
    t1 = time.time()
    #load graph structure info------
    if args.graph_format == 'adjlist':
        g.read_adjlist(path=args.graph_file, directed=args.directed)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(path=args.graph_file, weighted=args.weighted, directed=args.directed)
    #load node attribute info------
    is_ane = (args.method == 'abrw' or args.method == 'tadw' or args.method == 'attrpure' or args.method == 'attrcomb' or args.method == 'aane')
    if is_ane:
        assert args.attribute_file != ''
        g.read_node_attr(args.attribute_file)
    #load node label info------
    t2 = time.time()
    print(f'STEP1: end loading data; time cost: {(t2-t1):.2f}s')


    #---------------------------------------STEP2: prepare data----------------------------------------------------
    print('\nSTEP2: start preparing data for link pred task......')
    t1 = time.time()
    test_node_pairs=[]
    test_edge_labels=[]
    if args.task == 'lp' or args.task == 'lp_and_nc':
        edges_removed = g.remove_edge(ratio=args.link_remove)
        test_node_pairs, test_edge_labels = generate_edges_for_linkpred(graph=g, edges_removed=edges_removed, balance_ratio=1.0)
    t2 = time.time()
    print(f'STEP2: end preparing data; time cost: {(t2-t1):.2f}s')


    #-----------------------------------STEP3: upstream embedding task-------------------------------------------------
    print('\nSTEP3: start learning embeddings......')
    print(f'the graph: {args.graph_file}; \nthe model used: {args.method}; \
            \nthe # of edges used during embedding (edges maybe removed if lp task): {g.get_num_edges()}; \
            \nthe # of nodes: {g.get_num_nodes()}; \nthe # of isolated nodes: {g.get_num_isolates()}; \nis directed graph: {g.get_isdirected()}')
    t1 = time.time()
    model = None
    if args.method == 'abrw': 
        model = abrw.ABRW(graph=g, dim=args.dim, alpha=args.ABRW_alpha, topk=args.ABRW_topk, number_walks=args.number_walks,
                            walk_length=args.walk_length, window=args.window_size, workers=args.workers)
    elif args.method == 'aane':
        model = aane.AANE(graph=g, dim=args.dim, lambd=args.AANE_lamb, rho=args.AANE_rho, maxiter=args.AANE_maxiter, 
                            mode='comb') #mode: 'comb' struc and attri or 'pure' struc
    elif args.method == 'tadw':
        model = tadw.TADW(graph=g, dim=args.dim, lamb=args.TADW_lamb, maxiter=args.TADW_maxiter)
    elif args.method == 'attrpure':
        model = attrpure.ATTRPURE(graph=g, dim=args.dim, mode='pca')  #mode: pca or svd
    elif args.method == 'attrcomb':
        model = attrcomb.ATTRCOMB(graph=g, dim=args.dim, comb_with='deepwalk', number_walks=args.number_walks, walk_length=args.walk_length,
                                    window=args.window_size, workers=args.workers, comb_method=args.AttrComb_mode)  #comb_method: concat, elementwise-mean, elementwise-max
    elif args.method == 'deepwalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length, num_paths=args.number_walks, dim=args.dim,
                                 workers=args.workers, window=args.window_size, p=args.Node2Vec_p, q=args.Node2Vec_q)
    else:
        print('method not found...')
        exit(0)
    t2 = time.time()
    print(f'STEP3: end learning embeddings; time cost: {(t2-t1):.2f}s')

    if args.save_emb:
        model.save_embeddings(args.emb_file + time.strftime(' %Y%m%d-%H%M%S', time.localtime()))
        print(f'Save node embeddings in file: {args.emb_file}')


    #---------------------------------------STEP4: downstream task-----------------------------------------------
    print('\nSTEP4: start evaluating ......: ')
    t1 = time.time()
    vectors = model.vectors
    del model, g
    #------lp task
    if args.task == 'lp' or args.task == 'lp_and_nc':
        #X_test_lp, Y_test_lp = read_edge_label(args.label_file)  #if you want to load your own lp testing data
        print(f'Link Prediction task; the percentage of positive links for testing: {(args.link_remove*100):.2f}%'
                + ' (by default, also generate equal negative links for testing)')
        clf = lpClassifier(vectors=vectors)     #similarity/distance metric as clf; basically, lp is a binary clf probelm
        clf.evaluate(test_node_pairs, test_edge_labels)
    #------nc task
    if args.task == 'nc' or args.task == 'lp_and_nc':
        X, Y = read_node_label(args.label_file)
        print(f'Node Classification task; the percentage of labels for testing: {((1-args.label_reserved)*100):.2f}%')
        clf = ncClassifier(vectors=vectors, clf=LogisticRegression())   #use Logistic Regression as clf; we may choose SVM or more advanced ones
        clf.split_train_evaluate(X, Y, args.label_reserved)
    t2 = time.time()
    print(f'STEP4: end evaluating; time cost: {(t2-t1):.2f}s')


if __name__ == '__main__':
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    #random.seed(2018)
    #np.random.seed(2018)
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')

