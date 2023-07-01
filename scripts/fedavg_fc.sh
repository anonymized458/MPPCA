for trial in 1 2 3 4 5
do
    dir='../save_results_fc/fedavg/professional2/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_fedavg_fc.py --trial=$trial \
    --rounds=40 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --momentum=0.9 \
    --model=simple-fc \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_fc/' \
    --partition='professional2' \
    --alg='fedavg' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=3 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'_local_ep=50.txt'

done
