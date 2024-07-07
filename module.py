from torch import nn
from torch.utils.data import Dataset
import collections

class BlockSkipGramModel(nn.Module):
    def __init__(self, num_embedding=1024, embedding_dim=128):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_embedding)        
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class BlockSkipGramModelDataSet(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 0
    

def tokenizer():
    vocab = collections.OrderedDict()
    with open('vocab.txt', mode='r') as f:
        tokens = f.readlines()
    
    for index, token in enumerate(tokens):
        token = token.rstrip()
        vocab[token] = index

    return vocab

vocab = tokenizer()
print(vocab.get('<AIR>', vocab.get('<UNKNOWN>')))
