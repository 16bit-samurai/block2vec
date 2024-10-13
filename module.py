import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import collections
import mysql.connector


class BlockSkipGramModel(nn.Module):
    def __init__(self, num_embedding=1162, embedding_dim=128):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_embedding)        
        self.activation = nn.Softmax(dim=1)

        with torch.no_grad():
            self.embed.weight.uniform_(-1, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class BlockSkipGramModelDataSet(Dataset):
    def __init__(self, user, password) -> None:
        super().__init__()

        _, cursor = self.connect_to_db(user, password)
        cursor.execute("SELECT count(*) from blocks")
        data_nums = cursor.fetchone()[0]

        self.data_nums = data_nums
        self.user = user
        self.password = password

        self._init = False


    @staticmethod
    def connect_to_db(user, password):
        connection = mysql.connector.connect(
            host="127.0.0.1",
            port=3188,
            user=user,
            password=password,
            database="defaults",
            auth_plugin="mysql_native_password"
        )
        
        cursor = connection.cursor()
        cursor.execute('set global max_connections=1000')
        cursor.execute('set global max_allowed_packet=1048576000')

        return connection, cursor
    
    def _init_connection(self):
        if self._init == True:
            return

        connection, cursor = self.connect_to_db(self.user, self.password)

        self.connection = connection
        self.cursor = cursor
        
        self._init = True

    def __getitem__(self, index):
        self._init_connection()

        return (self.cursor, str(index))

        # cursor = self.cursor
        # # cursor.execute('SELECT center, target FROM blocks LIMIT 1 OFFSET {}'.format(index))
        # cursor.execute('SELECT center, target FROM blocks where id = {}'.format(index))
        # data, label = cursor.fetchone()
        # return torch.tensor(data, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.data_nums


    @staticmethod
    def collate_fn(batch_indices):
        batch_size = len(batch_indices)
        cursor = batch_indices[0][0]
        index_list = [index for _, index in batch_indices]
        cursor.execute('SELECT center, target FROM blocks where id in ({})'.format(','.join(index_list)))
        data, label = torch.tensor(cursor.fetchall()).split(split_size=1, dim=1)
        return data.squeeze(1), label.squeeze(1)
    

def tokenizer():
    vocab = collections.OrderedDict()
    with open('vocab.txt', mode='r') as f:
        tokens = f.readlines()
    
    for index, token in enumerate(tokens):
        token = token.rstrip()
        vocab[token] = index

    return vocab
    