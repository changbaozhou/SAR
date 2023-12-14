# bash run_resnet20.sh >> out_resnet20.log 2>&1 &
# resnet20 SGDcifar100
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SGD \
#        --dataset cifar10 \
#        --opt SGD \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # resnet20 SAM 0.01
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.01 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# resnet20 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# resnet20 SAM 0.08
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.08 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# resnet20 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1  \
#        --weight_decay 5e-4  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.15
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.15 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.2
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.2 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.5
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1 \
       --weight_decay 5e-4  \
       --rho 0.5 \
       --batch_size 128  \
       --arch resnet20 \
       --name resnet20_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine
# # # resnet20 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine
# # # resnet20 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine


# resnet20 SGDcifar100
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SGD \
#        --dataset cifar100 \
#        --opt SGD \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # resnet20 SAM 0.01
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.01 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# resnet20 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# resnet20 SAM 0.08
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.08 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# resnet20 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1  \
#        --weight_decay 5e-4  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.15
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.15 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.2
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.2 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # resnet20 SAM 0.5
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.5 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine
# # # resnet20 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine
# # # resnet20 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine