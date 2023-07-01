for trial in 1 2 3 4 5
do
    dir='../save_results_attack/mypca_delta_para/noniid-labeldir/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_mypca_attack.py --trial=$trial \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=0.5 \
    --bound=1 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_attack/' \
    --partition='noniid-labeldir' \
    --alg='mypca_delta_para' \
    --beta=0.5 \
    --local_view \
    --noise=0 \
    --gpu=3 \
    --print_freq=10 \
    --badnum=2 \
    --badrange=0.1 \
    2>&1 | tee $dir'/'$trial'_a=0.5_b=1.txt'

done
