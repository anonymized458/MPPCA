for trial in 1
do
    dir='../save_results_lf/l2/noniid-labeldir/fmnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_l2_lf_cnn.py --trial=$trial \
    --times=5 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=fmnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='l2' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=6 \
    --print_freq=10 \
    --flipnum=30 \
    --alpha=0.1 \
    2>&1 | tee $dir'/'$trial'_flipnum=30_beta=0.05.txt'

done
