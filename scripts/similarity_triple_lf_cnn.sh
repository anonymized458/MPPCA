for trial in 1 2 3 4 5
do
    dir='../save_results_triple_lf/similarity/noniid-labeldir/cifar100'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_similarity_triple_lf_cnn.py --trial=$trial \
    --rounds=80 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=1 \
    --momentum=0.9 \
    --model=simple-cnn-3 \
    --dataset=cifar100 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_triple_lf/' \
    --partition='noniid-labeldir' \
    --alg='similarity' \
    --beta=0.5 \
    --local_view \
    --noise=0 \
    --gpu=1 \
    --print_freq=10 \
    --flipnum=50 \
    2>&1 | tee $dir'/'$trial'_flipnum=50_beta=0.5.txt'

done
