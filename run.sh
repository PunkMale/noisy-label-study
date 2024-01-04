#!/bin/bash

# 设置噪声率列表
configs=('cifar10_vgg19' 'cifar10_res18' 'cifar10_res34' 'cifar10_dense121' 'cifar10_mobilev2')
#configs=('cifar100_res34' 'cifar100_res50' 'cifar100_res101' 'cifar100_res152')
noise_rates=(0.0 0.2 0.4 0.6 0.8)  # 0.0 0.2 0.4 0.6 0.8
gpu=2

for config in "${configs[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
        echo "config: $config, noise_rate: $noise_rate"
        python main.py \
            --config $config \
            --noise_rate $noise_rate \
            --gpu $gpu \
            --dataparallel \
            --eval_freq 1 \
            --tuning
    done
done
