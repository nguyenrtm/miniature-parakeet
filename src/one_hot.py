from typing import Union, Iterable
import torch
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator

class OneHotEncoder:
    def __init__(self, vocab: [str]):
        self.vocab = build_vocab_from_iterator(vocab)

    def one_hot(self, keys: Union[str, Iterable]):
        if isinstance(keys, str):
            keys = [keys]
        return F.one_hot(torch.tensor(self.vocab(keys)), num_classes=len(self.vocab)).squeeze().to(torch.float)