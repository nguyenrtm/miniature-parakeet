from spacy_features import SpacyFeatures
from one_hot import OneHotEncoder
import torch

class SentenceFeatureBuilder:
    def __init__(self,
                 spacy_features,
                 word_embedding_instance, 
                 padding_size: int = 200,
                 crop_in_between: int = 0):
        self.preprocesser = spacy_features
        self.we = word_embedding_instance 
        self.padding_size = padding_size
        self.crop_in_between = crop_in_between
        self.tag_labels = self.preprocesser.nlp.get_pipe("tagger").labels
        
    def get_position_embedding_given_ent(self, 
                                         ent_start: int, 
                                         ent_end: int, 
                                         text_length: int):
        '''
        Given entity index, get position embedding of sentence
        '''
        lst = []
        count_bef = ent_start
        count_in = ent_end - ent_start
        count_aft = text_length - ent_end

        for i in range(count_bef, 0, -1):
            lst.append(-i)
        
        for i in range(count_in + 1):
            lst.append(0)

        for i in range(1, count_aft + 1):
            lst.append(i)
        return lst

    def build_position_embedding(self, row):
        token_lst = self.get_tokens(row['text'])
        ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx = self.return_idx(row)
        text_length = token_lst[-1][2]
        pos_ent1 = torch.tensor(self.get_position_embedding_given_ent(ent1_start_idx, ent1_end_idx, text_length)).view(-1, 1)
        pos_ent2 = torch.tensor(self.get_position_embedding_given_ent(ent2_start_idx, ent2_end_idx, text_length)).view(-1, 1)
        zero_ent1 = torch.zeros((len(token_lst), 1))
        zero_ent1[ent1_start_idx] = 1.
        zero_ent1[ent1_end_idx] = 1.
        zero_ent2 = torch.zeros((len(token_lst), 1))
        zero_ent2[ent2_start_idx] = 1.
        zero_ent2[ent2_end_idx] = 1.
        to_return = torch.hstack((pos_ent1, pos_ent2, zero_ent1, zero_ent2))
        return to_return
    
    def get_tokens(self, text: str):
        return self.preprocesser.wordTokenizer(text)

    def return_idx(self, row):
        token_lst = self.get_tokens(row['text'])
        ent1_start_idx = self.find_idx_start_given_offset(token_lst, offset=row['ent1_start'])
        ent1_end_idx = self.find_idx_end_given_offset(token_lst, offset=row['ent1_end'])
        ent2_start_idx = self.find_idx_start_given_offset(token_lst, offset=row['ent2_start'])
        ent2_end_idx = self.find_idx_end_given_offset(token_lst, offset=row['ent2_end'])
        
        return ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx
                
    def find_idx_start_given_offset(self,
                                    token_lst, 
                                    offset):
        for i in range(len(token_lst) - 1):
            if offset >= token_lst[i][1] and offset < token_lst[i+1][1]:
                return token_lst[i][2]
        
        return len(token_lst) - 1
            
    def find_idx_end_given_offset(self, 
                                  token_lst, 
                                  offset):
        for i in range(len(token_lst) - 1):
            if offset >= token_lst[i][1] and offset - 1 < token_lst[i+1][1]:
                return token_lst[i][2]
            
        return len(token_lst) - 1

    def build_embedding(self, 
                        row):
        embedding_dictionary = dict()
        tag_lst = self.preprocesser.tag(row['text'])
        token_lst = self.get_tokens(row['text'])
        position = self.build_position_embedding(row)

        for i in range(len(token_lst)):
            embedding_dictionary[str(i)] = (token_lst[i][0], tag_lst[i], position[i])
            
        return embedding_dictionary