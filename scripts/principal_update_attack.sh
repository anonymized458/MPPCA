for trial in 1 2 3 4 5
do
    dir='../save_results_update_attack_fc2/principal/noniid-labeldir/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_principal_update_attack.py --trial=$trial \
    --rounds=40 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=5 \
    --momentum=0.9 \
    --model=simple-fc \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_update_attack_fc2/' \
    --partition='noniid-labeldir' \
    --alg='principal' \
    --beta=0.05 \
    --local_view \
    --noise=0 \
    --gpu=4 \
    --print_freq=10 \
    --badnum=2 \
    --badrange=0.4 \
    2>&1 | tee $dir'/'$trial'_a=5_beta=0.05_badrange=0.4.txt'

done
