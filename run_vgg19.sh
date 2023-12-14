# # bash run_vgg19.sh >> out_vgg19.log 2>&1 &
# # vgg19 SGD
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SGD \
#        --dataset cifar10 \
#        --opt SGD \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # vgg19 SAM 0.01
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.01 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # vgg19 SAM 0.05
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1 \
       --weight_decay 5e-4  \
       --rho 0.05 \
       --batch_size 128  \
       --arch vgg19 \
       --name vgg19_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine \
       --SCE_loss True \
# # vgg19 SAM 0.08
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1 \
       --weight_decay 5e-4  \
       --rho 0.08 \
       --batch_size 128  \
       --arch vgg19 \
       --name vgg19_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine \
       --SCE_loss True \
# # # vgg19 SAM 0.1
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1  \
       --weight_decay 5e-4  \
       --rho 0.1 \
       --batch_size 128  \
       --arch vgg19 \
       --name vgg19_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine
       --SCE_loss True \
# # # vgg19 SAM 0.15
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.15 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # # vgg19 SAM 0.2
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.2 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # # vgg19 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # # vgg19 SAM 0.5
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.5 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # # vgg19 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine
# # # vgg19 SAM 1.0
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1 \
       --weight_decay 5e-4  \
       --rho 1.0 \
       --batch_size 128  \
       --arch vgg19 \
       --name vgg19_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine

# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SGD \
#        --dataset cifar100 \
#        --opt SGD \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # vgg19 SAM 0.01
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.01 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # vgg19 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # vgg19 SAM 0.08
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.08 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # vgg19 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1  \
#        --weight_decay 5e-4  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine
#        --SCE_loss True \
# # # vgg19 SAM 0.15
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.15 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # vgg19 SAM 0.2
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.2 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # vgg19 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # vgg19 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.5 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # # vgg19 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine
# # # vgg19 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch vgg19 \
#        --name vgg19_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine