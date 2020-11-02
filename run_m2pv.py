import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
import sys

sys.path.append('./..')
sys.path.append('./../..')
from gensim.models import Word2Vec
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

metapaths = None
walk_length = 64
num_walks_per_node = 10
emb_dim = 128
model_use_data_DIR = None
model_save_path = None

def setup(
        dataset,
        _emb_dim=128,
        _model_save_path = None,
        _model_use_data_DIR = None
):
    global metapaths
    global emb_dim
    global model_save_path
    global model_use_data_DIR

    emb_dim = _emb_dim
    if dataset == 'dblp':
        metapaths = [
            ["A", "P", "A"],
            ["A", "P", "T", "P", "A"],
            ["A", "P", "C", "P", "A"],
            ["T", "P", "T"]
        ]

    if _model_save_path is None:
        model_save_path = 'model_save_data'
    else:
        model_save_path = _model_save_path
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    model_save_path = os.path.join(model_save_path, 'mp2vec')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    if _model_use_data_DIR is None:
        model_use_data_DIR = _model_use_data_DIR
    else:
        model_use_data_DIR = 'model_save_data'

    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    model_use_data_DIR = os.path.join(model_use_data_DIR, 'mp2vec')

    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)

    return
# ========================================== #

def generate_random_walks(graph_obj, num_walks_per_node, walk_length, metapaths):

    random_walk_object = UniformRandomMetaPathWalk(graph_obj)
    cpu_count = multiprocessing.cpu_count()
    list_nodes = list(graph_obj.nodes())
    num_chunks = cpu_count
    chunk_len = (len(list_nodes) // num_chunks)
    chunks = [
        list_nodes[i * chunk_len: (i + 1) * chunk_len] for i in range(0, num_chunks + 1)
    ]

    res = Parallel(n_jobs=cpu_count)(
        delayed(aux_gen_walks)(
            node_chunk, walk_length, random_walk_object, metapaths, num_walks_per_node
        )
        for node_chunk in chunks
    )

    all_walks = []
    for r in res:
        all_walks.extend(r)
    return all_walks

def aux_gen_walks(
        node_chunk,
        walk_length,
        random_walk_object,
        metapaths=None,
        num_walks=1
):
    if len(node_chunk) == 0:
        return []
    walks = random_walk_object.run(
        nodes=node_chunk,
        length=walk_length,
        n=num_walks,
        metapaths=metapaths
    )
    return walks

# ========================================
# Main function
# Input :
# Dataframe for each node type , with index col as node ids
# Dataframe for
# ========================================

def execute_model(
    dict_node_df,
    df_edges
):
    global dataset
    global model_save_path
    global metapaths
    global emb_dim
    global num_walks_per_node
    global walk_length
    print('Metapaths ', metapaths)

    emb_fpath = os.path.join(
        model_save_path,
        'n2v_{}_{}_{}.npy'.format(
            emb_dim, num_walks_per_node, walk_length)
    )
    graph_obj = StellarGraph(
        dict_node_df,
        df_edges
    )

    walks_save_file = "n2v_random_walks_{}_{}.npy".format(walk_length, num_walks_per_node)
    walks_save_file = os.path.join(model_use_data_DIR, walks_save_file)
    try:
        walks_np_arr = np.load(walks_save_file)
        walks = [list(_) for _ in walks_np_arr]
    except:
        walks = generate_random_walks(
            graph_obj, num_walks_per_node, walk_length, metapaths)
        walks_np_arr = np.array(walks)
        np.save(walks_save_file, walks_np_arr)

    print("Number of random walks: {}".format(len(walks)))
    str_walks = [[str(n) for n in walk] for walk in walks]

    if not os.path.exists(emb_fpath):

        word2vec_params = {
            'sg': 0,
            "size": emb_dim,
            "alpha": 0.5,
            "min_alpha": 0.001,
            'window': 5,
            'min_count': 0,
            "workers": multiprocessing.cpu_count(),
            "negative": 1,
            "hs": 0,  # 0: negative sampling, 1:hierarchical  softmax
            'compute_loss': True,
            'iter': 10,
            'cbow_mean': 1,
        }

        iters = 20
        mp2v_model = Word2Vec(**word2vec_params)
        mp2v_model.build_vocab(str_walks)
        losses = []
        learning_rate = 0.5
        step_size = (0.5 - 0.001) / iters

        for i in tqdm(range(iters)):
            trained_word_count, raw_word_count = mp2v_model.train(
                str_walks,
                compute_loss=True,
                start_alpha=learning_rate,
                end_alpha=learning_rate,
                total_examples=mp2v_model.corpus_count,
                epochs=1
            )
            loss = mp2v_model.get_latest_training_loss()
            losses.append(loss)
            print('>> ', i, ' Loss:: ', loss, learning_rate)
            learning_rate -= step_size

        # ======== Save node weights ============ #
        node_embeddings = []
        for i in range(len(graph_obj.nodes())):
            vec = mp2v_model.wv[str(i)]
            node_embeddings.append(vec)

        node_embeddings = np.array(node_embeddings)
        np.save(emb_fpath, node_embeddings)
    else:
        node_embeddings = np.load(emb_fpath)

    return node_embeddings
