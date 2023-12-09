import codecs
import torch
import numpy as np
from nltk.corpus import wordnet as wn
import pickle as pkl


def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data['embeddings']


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def load_vocab_utf8(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with codecs.open(filename, encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d

def load_pkl(path):
    with open(path, 'rb') as file:
        return pkl.load(file)
    
def dump_pkl(obj, path):
    with open(path, 'wb') as file:
        pkl.dump(obj, file)

def build_label_for_df(df):
    '''
    Build label for dataframe
    '''
    for i in range(len(df)):
        label = torch.tensor([df.iloc[i]['label']]).type(torch.FloatTensor)
        if i == 0:
            labels = label
        else:
            labels = torch.vstack((labels, label))
    
    return labels

def get_lookup(path):
    '''
    Get lookup table from file
    '''
    with open(path, 'r') as file:
        f = file.read().split('\n')
    return {f[i]: i + 1 for i in range(len(f))}

def lookup(word, dct):
    '''
    Get index of word in lookup table
    '''
    try: 
        idx = dct[word]
    except:
        idx = 0
    return idx

def get_idx(sent, vocab_lookup, tag_lookup, direction_lookup, edge_lookup):
    '''
    Get index of features of tokens in sentence
    '''
    if sent == None:
        return None
    to_return = list()
    i = 0
    for dp in sent:
        word1_idx = lookup(dp[0][0], vocab_lookup)
        word2_idx = lookup(dp[2][0], vocab_lookup)
        tag1_idx = lookup(dp[0][1], tag_lookup)
        tag2_idx = lookup(dp[2][1], tag_lookup)
        direction_idx = lookup(dp[1][0], direction_lookup)
        edge_idx = lookup(dp[1][1], edge_lookup)
        pos1 = dp[0][2]
        pos2 = dp[2][2]
        v = torch.tensor([word1_idx, tag1_idx, direction_idx, edge_idx, word2_idx, tag2_idx])
        v = torch.hstack((v[:2], pos1, v[2:6], pos2))
        if i == 0:
            to_return = v.view(1, -1)
        else:
            to_return = torch.vstack((to_return, v))
        i += 1
    return to_return

def get_idx_dataset(data,
                    vocab_lookup,
                    tag_lookup,
                    direction_lookup,
                    edge_lookup):
    tmp = list()
    for i in data:
        tmp.append(get_idx(i, vocab_lookup, tag_lookup, direction_lookup, edge_lookup))
    return tmp

def prepare_dict_simul_paths(lookup, X, y):
    '''
    Get training tensors from lookup table and X for simultaneous paths approach
    Args:
        lookup: lookup table
        X: list of tensors
    Returns:
        list of tensors
    '''
    to_return_dict = dict()

    for k, v in lookup.items():
        tmp = []
        for i in v:
            if X[i] != None:
                tmp.append(X[i])
            label = y[i]
        if tmp != []:
            to_return_dict[k] = (tmp, label)

    return to_return_dict
def get_tensors_simul_paths(lookup: dict,
                            select_option: str,
                            top_k: int):
    '''
    Get top k paths and convert to tensors for simultaneous paths approach
    Args:
        lookup: lookup table
        select_option: select method, could be "random", "shortest", "longest"
        top_k: number of paths to select
    Returns:
        X: dataset of tensors
        y: labels
    '''
    from tqdm import tqdm

    X = list()
    y = list()
    for k, v in tqdm(lookup.items()):
        X_tmp = v[0]
        y_tmp = v[1]
        if select_option == "random":
            if top_k < len(X_tmp):
                X_tmp = X_tmp[:top_k]
        X.append(X_tmp)
        y.append(y_tmp)

    return X, y