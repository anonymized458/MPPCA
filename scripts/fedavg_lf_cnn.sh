for trial in 1
do
    dir='../save_results_lf/fedavg/noniid-labeldir/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_fedavg_lf_cnn.py --trial=$trial \
    --times=5 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='fedavg' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=6 \
    --print_freq=10 \
    --flipnum=30 \
    2>&1 | tee $dir'/'$trial'_flipnum=30_beta=0.05.txt'

done
