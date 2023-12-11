import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from eval import evaluate_bc5

class SWCnn(nn.Module):
    def __init__(self,
                 we,
                 word_embedding_size: int = 300,
                 tag_number: int = 51,
                 tag_embedding_size: int = 20,
                 position_number: int = 4,
                 position_embedding_size: int = 20,
                 direction_number: int = 3,
                 direction_embedding_size: int = 5,
                 edge_number: int = 46,
                 edge_embedding_size: int = 20,
                 token_embedding_size: int = 500,
                 dep_embedding_size: int = 200,
                 conv1_out_channels: int = 16,
                 conv2_out_channels: int = 16,
                 conv3_out_channels: int = 16,
                 conv1_length: int = 1,
                 conv2_length: int = 2,
                 conv3_length: int = 3,
                 dropout_rate: float = 0.5,
                 ):

        super(SWCnn, self).__init__()

        self.w2v = nn.Embedding.from_pretrained(torch.tensor(we.vectors))
        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)
        self.direction_embedding = nn.Embedding(direction_number, direction_embedding_size, padding_idx=0)
        self.edge_embedding = nn.Embedding(edge_number, edge_embedding_size, padding_idx=0)

        self.normalize_position = nn.Linear(in_features=position_number,
                                             out_features=position_embedding_size,
                                             bias=False)
        
        self.normalize_tokens = nn.Linear(in_features=word_embedding_size+tag_embedding_size+position_embedding_size,
                                          out_features=token_embedding_size,
                                          bias=False)
        
        self.normalize_dep = nn.Linear(in_features=direction_embedding_size+edge_embedding_size,
                                       out_features=dep_embedding_size,
                                       bias=False)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv1_out_channels,
                      kernel_size=(conv1_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv2_out_channels,
                      kernel_size=(conv2_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv3_out_channels,
                      kernel_size=(conv3_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense_to_tag = nn.Linear(in_features=conv1_out_channels + conv2_out_channels + conv3_out_channels,
                                      out_features=2,
                                      bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        word_embedding_ent1 = self.w2v(x[:, :, 0].long())
        tag_embedding_ent1 = self.dropout(self.tag_embedding(x[:, :, 1].long()))
        position_embedding_ent1 = self.dropout(self.normalize_position(x[:, :, 2:6]))
        position_embedding_ent1 = self.relu(position_embedding_ent1)

        direction_embedding = self.dropout(self.direction_embedding(x[:, :, 6].long()))
        edge_embedding = self.dropout(self.edge_embedding(x[:, :, 7].long()))

        word_embedding_ent2 = self.w2v(x[:, :, 8].long())
        tag_embedding_ent2 = self.dropout(self.tag_embedding(x[:, :, 9].long()))
        position_embedding_ent2 = self.dropout(self.normalize_position(x[:, :, 10:14]))
        position_embedding_ent2 = self.relu(position_embedding_ent2)

        tokens_ent1 = torch.cat((word_embedding_ent1, tag_embedding_ent1, position_embedding_ent1), dim=2).float()
        tokens_ent2 = torch.cat((word_embedding_ent2, tag_embedding_ent2, position_embedding_ent2), dim=2).float()
        dep = torch.cat((direction_embedding, edge_embedding), dim=2).float()

        tokens_ent1 = self.dropout(self.normalize_tokens(tokens_ent1))
        tokens_ent1 = self.relu(tokens_ent1)
        tokens_ent2 = self.dropout(self.normalize_tokens(tokens_ent2))
        tokens_ent2 = self.relu(tokens_ent2)
        dep = self.dropout(self.normalize_dep(dep))
        dep = self.relu(dep)

        x = torch.cat((tokens_ent1, dep, tokens_ent2), dim=2)
        # print(x.shape)

        x = x.unsqueeze(1)
        # print(x.shape)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        # print(x1.shape, x2.shape, x3.shape)

        x1 = x1.squeeze(dim=3).permute(1, 0, 2)
        x2 = x2.squeeze(dim=3).permute(1, 0, 2)
        x3 = x3.squeeze(dim=3).permute(1, 0, 2)
        # print(x1.shape, x2.shape, x3.shape)

        x1 = x1.flatten(1, 2)
        x2 = x2.flatten(1, 2)
        x3 = x3.flatten(1, 2)
        # print(x1.shape, x2.shape, x3.shape)

        x1 = torch.max(x1, dim=1)[0]
        x2 = torch.max(x2, dim=1)[0]
        x3 = torch.max(x3, dim=1)[0]
        # print(x1.shape, x2.shape, x3.shape)

        x = torch.cat([x1, x2, x3], dim=0)
        # print(x.shape)
        x = self.dense_to_tag(x)
        # print(x.shape)
        x = self.softmax(x)
        # print(x.shape)

        return x

class Trainer:
    def __init__(self,
                 we,
                 lr: float = 0.0001,
                 weight_decay: float = 1e-4,
                 word_embedding_size: int = 300,
                 tag_number: int = 51,
                 tag_embedding_size: int = 50,
                 position_number: int = 4,
                 position_embedding_size: int = 50,
                 direction_number: int = 3,
                 direction_embedding_size: int = 10,
                 edge_number: int = 46,
                 edge_embedding_size: int = 50,
                 token_embedding_size: int = 500,
                 dep_embedding_size: int = 200,
                 conv1_out_channels: int = 512,
                 conv2_out_channels: int = 512,
                 conv3_out_channels: int = 256,
                 conv1_length: int = 1,
                 conv2_length: int = 2,
                 conv3_length: int = 3,
                 dropout_rate: float = 0.5,
                 device='cpu'):
        
        self.model = SWCnn(we,
                           word_embedding_size, 
                           tag_number,
                           tag_embedding_size, 
                           position_number,
                           position_embedding_size,
                           direction_number,
                           direction_embedding_size,
                           edge_number,
                           edge_embedding_size,
                           token_embedding_size,
                           dep_embedding_size,
                           conv1_out_channels,
                           conv2_out_channels,
                           conv3_out_channels,
                           conv1_length,
                           conv2_length,
                           conv3_length,
                           dropout_rate).to(device)
        
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=lr, 
                                    weight_decay=weight_decay)
        
        self.device = device
        self.train_loss_list = []
        self.val_loss_list = []
        self.p = []
        self.r = []
        self.f = []
        self.intra_p = []
        self.intra_r = []
        self.intra_f = []
        self.inter_p = []
        self.inter_r = []
        self.inter_f = []
        
    def convert_label_to_2d(self, label):
        if label.to(self.device) == torch.tensor([0]).to(self.device):
            tmp = torch.tensor([1., 0.]).to(self.device)
        else:
            tmp = torch.tensor([0., 1.]).to(self.device)
        
        return tmp
                

    def train_one_epoch(self, training_data):
        running_loss = 0.
        i = 0

        for data, label in training_data:
            data = data.to(self.device)
            label = self.convert_label_to_2d(label)
            i += 1
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    
    def validate(self, test_data, loss_list):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for data, label in test_data:
                data = data.to(self.device)
                label = label.to(self.device)
                outputs = self.model(data)
                label_for_loss = self.convert_label_to_2d(label)
                loss = self.criterion(outputs, label_for_loss)
                running_loss += loss.item()
                prediction = torch.argmax(outputs).unsqueeze(dim=0)
                predictions = torch.cat((predictions, prediction))
                labels = torch.cat((labels, label))

        f1 = BinaryF1Score().to(self.device)(predictions, labels)
        p = BinaryPrecision().to(self.device)(predictions, labels)
        r = BinaryRecall().to(self.device)(predictions, labels)
        
        loss_list.append([running_loss, f1.item(), p.item(), r.item()])
        return predictions
    
    def eval_bc5(self, pred, lookup):
        lst = self.convert_pred_to_lst(pred, lookup)
        return_tuple = evaluate_bc5(lst)
        self.p.append(return_tuple[0][0])
        self.r.append(return_tuple[0][1])
        self.f.append(return_tuple[0][2])
        self.intra_p.append(return_tuple[1][0])
        self.intra_r.append(return_tuple[1][1])
        self.intra_f.append(return_tuple[1][2])
        self.inter_p.append(return_tuple[2][0])
        self.inter_r.append(return_tuple[2][1])
        self.inter_f.append(return_tuple[2][2])
        return return_tuple

    
    def train(self, training_data, test_data, num_epochs):
        loss = list()

        for epoch in range(num_epochs):
            running_loss = self.train_one_epoch(training_data)
            loss.append(running_loss)
            print(f"Epoch {epoch + 1}")
            
            print("===== Validation =====")
            print("Training set:")
            pred_train = self.validate(training_data, self.train_loss_list)
            print(self.train_loss_list[-1])
            print("Test set:")
            pred_val = self.validate(test_data, self.val_loss_list)
            print(self.val_loss_list[-1])
        return pred_train, pred_val

    def convert_pred_to_lst(self, pred, lookup):
        lst = list()
        i = 0
        for k, v in lookup.items():
            if pred[i] == 1:
                abs_id = k[0]
                ent1_id = k[1]
                ent2_id = k[2]
                lst.append((abs_id, f"{ent1_id}_{ent2_id}", "CID"))
            i += 1

        return lst