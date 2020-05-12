# see https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-mixed.py

'''
sudo apt install python3-dev python3-pip virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate

git clone https://github.com/enijkamp/apex_example.git
cd apex_example
pip3 install -r requirements.txt

cd ..
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ../apex_example
python3 test_apex_distributed_spawn.py
'''

import os
from datetime import datetime
import random
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import apex.parallel
from apex import amp


def main():
    spawn_workers(parse_args())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')

    parser.add_argument('--apex_enabled', default=True, type=bool, help='enable apex')
    parser.add_argument('--apex_opt_level', default='O2', type=str, help='apex optimization level (O0, O1, O2, O3)')

    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=100, type=int, metavar='N', help='batch size')
    return parser.parse_args()


def spawn_workers(args):
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_gpu(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ConvNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    # distributed
    rank = args.nr * args.gpus + gpu
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # preamble
    set_cuda(deterministic=True)
    set_seed(seed=0)
    set_gpu(gpu)

    # model
    model = ConvNet().cuda(gpu)
    model.cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    if args.apex_enabled:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_opt_level)
        model = apex.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # data
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    # train
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            if args.apex_enabled:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0 and gpu == 0:
                print(f'Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    if gpu == 0:
        print(f'Training complete in: {datetime.now() - start}')


if __name__ == '__main__':
    main()
