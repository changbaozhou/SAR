# bash run_wideresnet28.sh >> out_wideresnet28.log 2>&1 &
# # wideresnet28 SGD
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SGD \
#        --dataset cifar10 \
#        --opt SGD \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.5
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1 \
       --weight_decay 1e-3  \
       --rho 0.5 \
       --batch_size 128  \
       --arch wideresnet28 \
       --name wideresnet28_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine \
       --SCE_loss True \
# # wideresnet28 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine \
#        --SCE_loss True \

# wideresnet28 SGD
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SGD \
#        --dataset cifar100\
#        --opt SGD \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# wideresnet28 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 0.5
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.1 \
       --weight_decay 1e-3  \
       --rho 0.5 \
       --batch_size 128  \
       --arch wideresnet28 \
       --name wideresnet28_SAM \
       --dataset cifar100 \
       --lr_scheduler cosine \
       --SCE_loss True \
# # wideresnet28 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \
# # wideresnet28 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 1e-3  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch wideresnet28 \
#        --name wideresnet28_SAM \
#        --dataset cifar100 \
#        --lr_scheduler cosine \
#        --SCE_loss True \