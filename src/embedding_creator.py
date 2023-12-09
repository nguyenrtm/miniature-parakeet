from typing import Union, Iterable
import spacy
import torch

from one_hot import OneHotEncoder
from utils import get_trimmed_w2v_vectors, load_vocab
from sentence_feature_builder import SentenceFeatureBuilder

class WordEmbedding:
    def __init__(self, path='../cache/w2v/biocreative_fasttext_pm.npz'):
        self.vectors = get_trimmed_w2v_vectors(path)
        self.vocab = load_vocab('../cache/w2v/all_words.txt')

    def get_word_vector(self, word):
        if word not in self.vocab.keys():
            return torch.tensor(self.vectors[-1]) # $UNK$ vector
        else:
            return torch.tensor(self.vectors[self.vocab[word]])
 

class EdgeEmbedding:
    def __init__(self, nlp):
        self.nlp = nlp
        self.labels = list(self.nlp.get_pipe("parser").labels)

    def one_hot(self, keys: Union[str, Iterable]):
        return self.one_hot_encoder.one_hot(keys)
    
    def get_direction(self, direction):
        assert direction in ['forward', 'reverse'], "direction should be forward or reverse"
        if direction == 'forward':
            return torch.tensor([1])
        elif direction == 'reverse':
            return torch.tensor([-1])
        
    def edge_to_tensor(self, edge):
        direction, dep = edge[1][0], edge[1][1]
        return torch.cat((self.get_direction(direction), self.one_hot(dep)))
    
    def path_with_dep_to_tensor(self, path_with_dep):
        to_return = torch.empty((0, 49))
        
        for edge in path_with_dep:
            tensor_tmp = self.edge_to_tensor(edge)
            to_return = torch.vstack((to_return, tensor_tmp))
        
        return to_return