for trial in 1
do
    dir='../save_results_noise/simplePCA/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_simplePCA_noise.py --trial=$trial \
    --times=5 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=-0.5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_noise/' \
    --partition='noniid-labeldir' \
    --alg='simplePCA' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --noise2=0.5 \
    --gpu=0 \
    --print_freq=10 \
    --flipnum=40 \
    2>&1 | tee $dir'/'$trial'_num=40_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_noise/simplePCA/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_simplePCA_noise.py --trial=$trial \
    --times=5 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=-0.5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_noise/' \
    --partition='noniid-labeldir' \
    --alg='simplePCA' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --noise2=0.5 \
    --gpu=0 \
    --print_freq=10 \
    --flipnum=50 \
    2>&1 | tee $dir'/'$trial'_num=50_beta=0.05.txt'

done





