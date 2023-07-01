for trial in 1
do
    dir='../save_results/mypca_cfl/noniid-#label2/cifar10'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_mypca_cfl --trial=$trial \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=cifar10 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results/' \
    --partition='noniid-#label2' \
    --alg='mypca_cfl' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=1 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'.txt'

done