# %%
from module import BlockSkipGramModelDataSet, BlockSkipGramModel
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import os
# %%
device = torch.device('cuda:0')

writer = SummaryWriter()
dataset = BlockSkipGramModelDataSet(os.environ["B2V_USER"], os.environ["B2V_PASSWD"])
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=BlockSkipGramModelDataSet.collate_fn, num_workers=4)
model = BlockSkipGramModel().to(device)


# %%
model.train()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for i, (data, label) in enumerate(loader):
    data = data.to(device)
    label = label.to(device)
    output = model(data)
    optimizer.zero_grad()
    loss = loss_func(F.one_hot(label, num_classes=1162).to(torch.float32), output)
    loss.backward()
    optimizer.step()
    if i % 10000 == 0:
        writer.add_scalar('train/loss', loss, global_step=i // 10000)

    
# %%
