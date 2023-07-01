for trial in 1
do
    dir='../save_results_lf/complex/noniid-labeldir/fmnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_complex_lf_cnn.py --trial=$trial \
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
    --dataset=fmnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='complex' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=7 \
    --print_freq=10 \
    --flipnum=30 \
    --sigma=0.5 \
    2>&1 | tee $dir'/'$trial'_sigma=0.5_flipnum=30_beta=0.05.txt'

done &
for trial in 1
do
    dir='../save_results_lf/complex/noniid-labeldir/fmnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_complex_lf_cnn.py --trial=$trial \
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
    --dataset=fmnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='complex' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=7 \
    --print_freq=10 \
    --flipnum=30 \
    --sigma=1 \
    2>&1 | tee $dir'/'$trial'_sigma=1_flipnum=30_beta=0.05.txt'

done
