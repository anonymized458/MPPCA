for trial in 1 2 3 4 5
do
    dir='../save_results_attack/principal/homo/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_principal_attack.py --trial=$trial \
    --rounds=100 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=5 \
    --momentum=0.9 \
    --model=simple-cnn \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_attack/' \
    --partition='homo' \
    --alg='principal' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=4 \
    --print_freq=10 \
    --badnum=2 \
    --badrange=0.1 \
    2>&1 | tee $dir'/'$trial'_a=5.txt'

done
