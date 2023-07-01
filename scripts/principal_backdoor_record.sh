for trial in 1
do
    dir='../save_results_backdoor/principal/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_principal_backdoor_record_weight.py --trial=$trial \
    --times=5 \
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
    --savedir='../save_results_backdoor/' \
    --partition='noniid-labeldir' \
    --alg='principal' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=7 \
    --print_freq=10 \
    --flipnum=20 \
    --backdoorfrac=1 \
    2>&1 | tee $dir'/'$trial'_flipnum=20_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_backdoor/principal/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_principal_backdoor_record_weight.py --trial=$trial \
    --times=5 \
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
    --savedir='../save_results_backdoor/' \
    --partition='noniid-labeldir' \
    --alg='principal' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=7 \
    --print_freq=10 \
    --flipnum=30 \
    --backdoorfrac=1 \
    2>&1 | tee $dir'/'$trial'_flipnum=30_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_backdoor/principal/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_principal_backdoor_record_weight.py --trial=$trial \
    --times=5 \
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
    --savedir='../save_results_backdoor/' \
    --partition='noniid-labeldir' \
    --alg='principal' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=7 \
    --print_freq=10 \
    --flipnum=40 \
    --backdoorfrac=1 \
    2>&1 | tee $dir'/'$trial'_flipnum=40_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_backdoor/principal/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_principal_backdoor_record_weight.py --trial=$trial \
    --times=5 \
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
    --savedir='../save_results_backdoor/' \
    --partition='noniid-labeldir' \
    --alg='principal' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=7 \
    --print_freq=10 \
    --flipnum=50 \
    --backdoorfrac=1 \
    2>&1 | tee $dir'/'$trial'_flipnum=50_beta=0.05.txt'

done
