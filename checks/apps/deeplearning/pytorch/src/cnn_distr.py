import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models
#from pt_distr_env import DistributedEnviron


num_warmup_epochs = 2
num_epochs = 5
batch_size_per_gpu = 256
num_iters = 25
model_name = 'resnet152'

os.environ['NCCL_NET_GDR_LEVEL'] = '2'

#distr_env = DistributedEnviron()
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
#device = int(os.environ['SLURM_LOCALID'])

#model = getattr(models, model_name)()
#model.to(device)
#model = model.cuda()

model = models.resnet152()
#device_id = int(os.environ['ROCR_VISIBLE_DEVICES'])
device = torch.device("cuda:0")
model.to(device)

ddp_model = DistributedDataParallel(model)


optimizer = optim.SGD(model.parameters(), lr=0.01)

class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.randn(3, 224, 224)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return batch_size_per_gpu * num_iters * world_size


#ddp_model = DistributedDataParallel(model, device_ids=[device])

train_set = SyntheticDataset()
train_sampler = DistributedSampler(
    train_set,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    seed=42
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    sampler=train_sampler,
    # num_workers=16
)


def benchmark_step(model, imgs, labels):
    optimizer.zero_grad()
    output = model(imgs.to(device))
    loss = F.cross_entropy(output, labels.to(device))
    loss.backward()
    optimizer.step()


# warmup
for epoch in range(num_warmup_epochs):
    for step, (imgs, labels) in enumerate(train_loader):
        benchmark_step(ddp_model, imgs, labels)

# benchmark
imgs_sec = []
for epoch in range(num_epochs):
    t0 = time.time()
    for step, (imgs, labels) in enumerate(train_loader):
        benchmark_step(ddp_model, imgs, labels)

    dt = time.time() - t0
    imgs_sec.append(batch_size_per_gpu * num_iters / dt)

    if rank == 0:
        print(f' * Rank {rank} - Epoch {epoch:2d}: '
              f'{imgs_sec[epoch]:.2f} images/sec per GPU')

imgs_sec_total = np.mean(imgs_sec) * world_size
if rank == 0:
    print(f' * Total average: {imgs_sec_total:.2f} images/sec')
