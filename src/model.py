import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

from eval import evaluate_bc5

class Model(nn.Module):
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
                 conv1_length: int = 2,
                 conv2_length: int = 3,
                 conv3_length: int = 4
                 ):

        super(Model, self).__init__()

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
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv2_out_channels,
                      kernel_size=(conv2_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=conv3_out_channels,
                      kernel_size=(conv3_length, token_embedding_size * 2 + dep_embedding_size),
                      stride=1,
                      bias=False),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.dense_to_tag = nn.Linear(in_features=conv1_out_channels + conv2_out_channels + conv3_out_channels,
                                      out_features=2,
                                      bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        word_embedding_ent1 = self.w2v(x[:, :, 0].long())
        tag_embedding_ent1 = self.tag_embedding(x[:, :, 1].long())
        position_embedding_ent1 = self.normalize_position(x[:, :, 2:6])
        position_embedding_ent1 = self.relu(position_embedding_ent1)

        direction_embedding = self.direction_embedding(x[:, :, 6].long())
        edge_embedding = self.edge_embedding(x[:, :, 7].long())

        word_embedding_ent2 = self.w2v(x[:, :, 8].long())
        tag_embedding_ent2 = self.tag_embedding(x[:, :, 9].long())
        position_embedding_ent2 = self.normalize_position(x[:, :, 10:14])
        position_embedding_ent2 = self.relu(position_embedding_ent2)

        tokens_ent1 = torch.cat((word_embedding_ent1, tag_embedding_ent1, position_embedding_ent1), dim=2).float()
        tokens_ent2 = torch.cat((word_embedding_ent2, tag_embedding_ent2, position_embedding_ent2), dim=2).float()
        dep = torch.cat((direction_embedding, edge_embedding), dim=2).float()

        tokens_ent1 = self.normalize_tokens(tokens_ent1)
        tokens_ent2 = self.normalize_tokens(tokens_ent2)
        dep = self.normalize_dep(dep)

        x = torch.cat((tokens_ent1, dep, tokens_ent2), dim=2)

        x = x.unsqueeze(1)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x1 = torch.max(x1.squeeze(dim=3), dim=2)[0]
        x2 = torch.max(x2.squeeze(dim=3), dim=2)[0]
        x3 = torch.max(x3.squeeze(dim=3), dim=2)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dense_to_tag(x)
        x = self.softmax(x)

        return x

class Trainer:
    def __init__(self,
                 we,
                 lr,
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
                 conv1_length: int = 2,
                 conv2_length: int = 3,
                 conv3_length: int = 4,
                 device='cpu'):
        
        self.model = Model(we,
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
                           conv3_length).to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
        
    def convert_label_to_2d(self, batch_label):
        i = 0
        for label in batch_label:
            i += 1
            if label == torch.tensor([0]).to(self.device):
                tmp = torch.tensor([1., 0.]).to(self.device)
            else:
                tmp = torch.tensor([0., 1.]).to(self.device)
            
            if i == 1:
                to_return = tmp.unsqueeze(0)
            else:
                to_return = torch.vstack((to_return, tmp))
        
        return to_return
                

    def train_one_epoch(self, training_loader):
        running_loss = 0.
        i = 0

        for batch_data, batch_label in training_loader:
            batch_data = torch.tensor(batch_data).to(self.device)
            batch_label = torch.tensor(batch_label).to(self.device)
            batch_label = self.convert_label_to_2d(batch_label)
            i += 1
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    
    def validate(self, validation_loader, loss_list):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for batch_data, batch_label in validation_loader:
                batch_data = torch.tensor(batch_data).to(self.device)
                batch_label = torch.tensor(batch_label).to(self.device)
                outputs = self.model(batch_data)
                batch_label_for_loss = self.convert_label_to_2d(batch_label)
                loss = self.criterion(outputs, batch_label_for_loss)
                running_loss += loss.item()
                
                batch_prediction = torch.argmax(outputs, dim=1)
                predictions = torch.cat((predictions, batch_prediction))
                labels = torch.cat((labels, batch_label))
        
        labels = labels.squeeze()
        true_pred = []
        for i in range(len(labels)):
            if labels[i].cpu() == 1:
                true_pred.append(i)

        f1 = BinaryF1Score().to(self.device)(predictions, labels)
        p = BinaryPrecision().to(self.device)(predictions, labels)
        r = BinaryRecall().to(self.device)(predictions, labels)
        
        loss_list.append([running_loss, f1.item(), p.item(), r.item()])
        return predictions
    
    def eval_bc5(self, pred, df):
        dct, lst = self.convert_pred_to_lst(pred, df)
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

    
    def train(self, training_loader, validation_loader, num_epochs):
        loss = list()

        for epoch in range(num_epochs):
            running_loss = self.train_one_epoch(training_loader)
            loss.append(running_loss)
            print(f"Epoch {epoch + 1}")
            
            print("===== Validation =====")
            print("Training set:")
            pred_train = self.validate(training_loader, self.train_loss_list)
            print(self.train_loss_list[-1])
            print("Validation set:")
            pred_val = self.validate(validation_loader, self.val_loss_list)
            print(self.val_loss_list[-1])
        return pred_train, pred_val

    def convert_pred_to_lst(self, pred, df):
        dct = {}
        for i in range(len(pred)):
            if pred[i] == 1:
                if df.iloc[i]['ent1_id'] == '-1' or df.iloc[i]['ent2_id'] == '-1':
                    continue
                elif len(df.iloc[i]['ent1_id'].split('|')) > 1:
                    tmp = df.iloc[i]['ent1_id'].split('|')
                    for ent in tmp:
                        idx = df.iloc[i]['sent_id'].split('_')[0]
                        if idx in dct.keys():
                            if f"{ent}_{df.iloc[i]['ent2_id']}" not in dct[idx]:
                                dct[idx].append(f"{ent}_{df.iloc[i]['ent2_id']}")
                        else:
                            dct[idx] = [f"{ent}_{df.iloc[i]['ent2_id']}"]
                elif len(df.iloc[i]['ent2_id'].split('|')) > 1:
                    tmp = df.iloc[i]['ent2_id'].split('|')
                    for ent in tmp:
                        idx = df.iloc[i]['sent_id'].split('_')[0]
                        if idx in dct.keys():
                            if f"{df.iloc[i]['ent1_id']}_{ent}" not in dct[idx]:
                                dct[idx].append(f"{df.iloc[i]['ent1_id']}_{ent}")
                        else:
                            dct[idx] = [f"{df.iloc[i]['ent1_id']}_{ent}"]
                else:
                    idx = df.iloc[i]['sent_id'].split('_')[0]
                    if idx in dct.keys():
                        if f"{df.iloc[i]['ent1_id']}_{df.iloc[i]['ent2_id']}" not in dct[idx]:
                            dct[idx].append(f"{df.iloc[i]['ent1_id']}_{df.iloc[i]['ent2_id']}")
                    else:
                        dct[idx] = [f"{df.iloc[i]['ent1_id']}_{df.iloc[i]['ent2_id']}"]

        lst = []
        for k, v in dct.items():
            for _ in v:
                lst.append((k, _, "CID"))

        return dct, lst