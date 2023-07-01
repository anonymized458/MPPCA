for trial in 1
do
    dir='../save_results_noise/l2/noniid-labeldir/fmnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_l2_noise.py --trial=$trial \
    --times=5 \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=0.1 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=fmnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_noise/' \
    --partition='noniid-labeldir' \
    --alg='l2' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --noise2=0.5 \
    --gpu=5 \
    --print_freq=10 \
    --flipnum=0 \
    2>&1 | tee $dir'/'$trial'_num=0_beta=0.05.txt'

done
