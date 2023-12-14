# bash run_pyramidnet110.sh >> out_pyramidnet110.log 2>&1 &
# # pyramidnet110 SGD
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SGD \
#        --dataset cifar10 \
#        --opt SGD \
#        --epochs 300
# # pyramidnet110 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar10 \
#        --epochs 300
# # pyramidnet110 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar10 \
#        --epochs 300
# # pyramidnet110 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar10 \
#        --epochs 300
# # pyramidnet110 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar10 \
#        --epochs 300
# # pyramidnet110 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar10 \
#        --epochs 300


# pyramidnet110 SGD
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SGD \
#        --dataset cifar100 \
#        --opt SGD \
#        --epochs 300
# pyramidnet110 SAM 0.05
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 0.05 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar100 \
#        --epochs 300
# pyramidnet110 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.05 \
#        --weight_decay 5e-4  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch pyramidnet110 \
#        --name pyramidnet110_SAM \
#        --dataset cifar100 \
#        --epochs 300
# pyramidnet110 SAM 0.25
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.05 \
       --weight_decay 5e-4  \
       --rho 0.25 \
       --batch_size 128  \
       --arch pyramidnet110 \
       --name pyramidnet110_SAM \
       --dataset cifar100 \
       --epochs 300
# pyramidnet110 SAM 0.8
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.05 \
       --weight_decay 5e-4  \
       --rho 0.8 \
       --batch_size 128  \
       --arch pyramidnet110 \
       --name pyramidnet110_SAM \
       --dataset cifar100 \
       --epochs 300
# pyramidnet110 SAM 1.0
python -m torch.distributed.launch --nproc_per_node=2 train.py \
       --learning_rate 0.05 \
       --weight_decay 5e-4  \
       --rho 1.0 \
       --batch_size 128  \
       --arch pyramidnet110 \
       --name pyramidnet110_SAM \
       --dataset cifar100 \
       --epochs 300