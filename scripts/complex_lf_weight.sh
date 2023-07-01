for trial in 1
do
    dir='../save_results_lf/complex/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_complex_lf_cnn_weight.py --trial=$trial \
    --times=2 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='complex' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    --flipnum=20 \
    --sigma=0.5 \
    2>&1 | tee $dir'/'$trial'_flipnum=20_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_lf/complex/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_complex_lf_cnn_weight.py --trial=$trial \
    --times=2 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='complex' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    --flipnum=30 \
    --sigma=0.5 \
    2>&1 | tee $dir'/'$trial'_flipnum=30_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_lf/complex/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_complex_lf_cnn_weight.py --trial=$trial \
    --times=2 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='complex' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    --flipnum=40 \
    --sigma=0.5 \
    2>&1 | tee $dir'/'$trial'_flipnum=40_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_lf/complex/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_complex_lf_cnn_weight.py --trial=$trial \
    --times=2 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='complex' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    --flipnum=50 \
    --sigma=0.5 \
    2>&1 | tee $dir'/'$trial'_flipnum=50_beta=0.05.txt'

done
