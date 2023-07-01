for trial in 1 2 3 4 5
do
    dir='../save_results_attack/fedavg/noniid-labeldir/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_fedavg_attack.py --trial=$trial \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.001 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_attack/' \
    --partition='noniid-labeldir' \
    --alg='fedavg' \
    --beta=0.5 \
    --local_view \
    --noise=0 \
    --gpu=3 \
    --print_freq=10 \
    --badnum=2 \
    --badrange=0.1 \
    2>&1 | tee $dir'/'$trial'_range=0.1.txt'

done
