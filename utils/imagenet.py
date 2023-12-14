import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import numpy as np
from utils.cutout import Cutout
import timm

from timm.data.dataset import ImageDataset
from timm.data.dataset_factory import create_dataset
from timm.data import create_loader


class ImageNet:
    def __init__(self, args):
        # if args.local_rank not in [-1, 0]:
            # torch.distributed.barrier()
        batch_size = args.batch_size
        threads = args.threads
        # mean, std = np.array([125.3, 123.0, 113.9]) / 255.0,np.array([63.0, 62.1, 66.7]) / 255.0

        # train_transform = transforms.Compose([
        #     torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        #     Cutout()
        # ])

        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])

        # train_set = torchvision.datasets.CIFAR10(root='/home/bobzhou/dataset', train=True, download=True, transform=train_transform) 
        # test_set = torchvision.datasets.CIFAR10(root='/home/bobzhou/dataset', train=False, download=True, transform=test_transform) if args.local_rank in [-1, 0] else None

        # # if args.local_rank == 0:
        #     # torch.distributed.barrier()

        # train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
 
        # self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=threads,sampler=train_sampler)
        # self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



        # Define data configuration for ResNet-18
        input_size = 224
        data_config = {'input_size': input_size}

        # Data transforms specific to the model
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the ImageNet dataset using timm's create_loader
        dataset = ImageDataset('/home/bobzhou/dataset',)
    
        train_loader = create_loader(
            root='/home/bobzhou/dataset',
            split='train',
            is_training=True,
            batch_size=batch_size,
            transform=transform,
            **data_config,
        )

        val_loader = create_loader(
            root='/home/bobzhou/dataset',
            split='val',
            is_training=False,
            batch_size=batch_size,
            transform=transform,
            **data_config,
        )
        
        self.train = train_loader
        self.test = val_loader