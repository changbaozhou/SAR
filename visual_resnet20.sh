# bash run_resnet20.sh >> out_resnet20.log 2>&1 &
# resnet20 SGD
# python -m torch.distributed.launch --nproc_per_node=2 visual.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SGD \
#        --dataset cifar10 \
#        --opt SGD \
#        --lr_scheduler cosine \
#        --resume_name SGD \
#        --output_dir /home/bobzhou/SAR/output
# # resnet20 SAM 0.05
python -m torch.distributed.launch --nproc_per_node=2 visual.py \
       --learning_rate 0.1 \
       --weight_decay 5e-4  \
       --rho 0.05 \
       --batch_size 128  \
       --arch resnet20 \
       --name resnet20_SAM \
       --dataset cifar10 \
       --lr_scheduler cosine \
       --resume_name SAM \
       --output_dir /home/bobzhou/SAR/output        
# # resnet20 SAM 0.1
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1  \
#        --weight_decay 5e-4  \
#        --rho 0.1 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine
# # resnet20 SAM 0.25
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.25 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine
# # resnet20 SAM 0.8
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 0.8 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine
# # resnet20 SAM 1.0
# python -m torch.distributed.launch --nproc_per_node=2 train.py \
#        --learning_rate 0.1 \
#        --weight_decay 5e-4  \
#        --rho 1.0 \
#        --batch_size 128  \
#        --arch resnet20 \
#        --name resnet20_SAM \
#        --dataset cifar10 \
#        --lr_scheduler cosine