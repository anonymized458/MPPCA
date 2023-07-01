for trial in 1 2 3 4 5
do
    dir='../save_results_lf/mypca_delta_para/noniid-labeldir/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_mypca_lf.py --trial=$trial \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=1 \
    --bound=5 \
    --momentum=0.9 \
    --model=simple-fc \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_lf/' \
    --partition='noniid-labeldir' \
    --alg='mypca_delta_para' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=6 \
    --print_freq=10 \
    --flipnum=50 \
    2>&1 | tee $dir'/'$trial'_triple_flipnum=50.txt'

done
