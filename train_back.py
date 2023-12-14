import os, sys, random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import time
from datetime import timedelta
import logging

from tensorboardX import SummaryWriter
import wandb


from tools.utils import *
from tools.initialize import initialize
from tools.options import get_args
from tools.log import Log
from tools.step_lr import lr_schedule
from tools.smooth_cross_entropy import smooth_crossentropy,trades_loss

# from model import PreActResNet18
# from models.PyramidNet import PyramidNet
# from models.wideresnet import WideResNet28
from models.CifarResNet import vanilla_resnet20

from tools.sam import SAM
from torch.nn.parallel import DistributedDataParallel as DDP



logger = logging.getLogger(__name__)

args = get_args()

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def train(args,model):
    # initialize
    initialize(args, seed=args.manualSeed)

    # initialize logger
    log = Log(log_each=10)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    if args.local_rank != -1:
        train_loader, test_loader = load_dataset_parallel(args, dataset, args.batch_size)
    else:
        train_loader, test_loader = load_dataset(dataset, args.batch_size)
    
    

    
    params = model.parameters()
    if args.SCE_loss =="True":
        criterion = smooth_crossentropy
    else:
        criterion = nn.CrossEntropyLoss()
    
    if args.opt == 'SGD': 
        opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'SAM':
        base_opt = torch.optim.SGD
        opt = SAM(params, base_opt,lr=args.max_lr, momentum=0.9, weight_decay=5e-4, rho=args.rho)
    normalize = normalize_cifar if dataset == 'cifar10' else normalize_cifar100

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model,device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    
    best_test_acc = 0
    # train
    for epoch in range(args.epochs):
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        start_time = time()
        log_data = [0,0,0,0] # train_loss, train_acc, test_loss, test_acc
        # train
        model.train()
        log.train(len_dataset=len(train_loader))

        lr = lr_schedule(args, epoch)
        opt.param_groups[0].update(lr=lr)
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)

            output = model(normalize(x))
            loss = criterion(output, y)
            
            if args.opt == 'SGD':
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            elif args.opt == 'SAM':
                loss.backward()
                opt.first_step(zero_grad=True)
                output_2 = model(normalize(x))
                criterion(output_2, y).backward()
                opt.second_step(zero_grad=True)
            
            
            log_data[0] += (loss * len(y)).item() #train_loss
            log_data[1] += (output.max(1)[1] == y).float().sum().item() #train_acc

            
        # test
        model.eval()
        for x, y in test_loader:
            
            x, y = x.to(args.device), y.to(args.device)
            output = model(normalize(x)).detach()
            loss = criterion(output, y)
            
            log_data[2] += (loss * len(y)).item() #test_loss
            log_data[3] += (output.max(1)[1] == y).float().sum().item() #test_acc
    
        
        log_data = np.array(log_data)
        log_data[0] /= 60000
        log_data[1] /= 60000
        log_data[2] /= 10000
        log_data[3] /= 10000

        # log and visualization
        if  args.local_rank in [-1, 0]:
            writer.add_scalar('loss/train_loss', log_data[0], epoch)
            writer.add_scalar('loss/test_loss', log_data[2], epoch)
            writer.add_scalar('accuracy/train_accuracy', log_data[1], epoch)
            writer.add_scalar('accuracy/test_accuracy', log_data[3], epoch)

        # save the best model
        if log_data[3] > best_test_acc:
            best_test_acc = log_data[3]
            torch.save(model.state_dict(), f'checkpoint/{args.fname}.pth')


def main(args):
   

    # Setup logging
    train_name = "train" 
    log_path = args.fname + "_" + train_name
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',filename = '../output/logs/'+log_path,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))


    if not args.wandb_id:  #如果没有输入就重新生成
        args.wandb_id = wandb.util.generate_id()
    wandb.init(
                project = "SAR",
                config = args,
                name = 'ResNet20_cifar10_SAM_rho0.8',
                # name = 'ResNet20_cifar10_SGD',
                sync_tensorboard=True,
                #resume = True,
                )
    
    
    # device = f'cuda:{args.device}'

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    
    args.device = device



    # init dataset
    dataset = args.dataset
    # set model 
    # model = PreActResNet18(10 if dataset == 'cifar10' else 100).to(device)
    model = vanilla_resnet20(10 if dataset == 'cifar10' else 100).to(device)
    # model = vanilla_resnet32(10 if dataset == 'cifar10' else 100).to(device)
    # model = PyramidNet(dataset,110,270,10 if dataset == 'cifar10' else 100,False).to(device)
    # model = WideResNet28(10 if dataset == 'cifar10' else 100).to(device)

    # train
    train(args, model)



    


if __name__ == '__main__':
    main(args)
        

    
