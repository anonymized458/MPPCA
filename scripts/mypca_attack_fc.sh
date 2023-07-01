for trial in 1 2 3 4 5
do
    dir='../save_results_attack_fc/mypca_delta_para/professional1/mnist'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi

    python ../main_mypca_attack_fc.py --trial=$trial \
    --rounds=40 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=50 \
    --local_bs=10 \
    --lr=0.001 \
    --alpha=1 \
    --bound=1 \
    --momentum=0.9 \
    --model=simple-fc \
    --dataset=mnist \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results_attack_fc/' \
    --partition='professional1' \
    --alg='mypca_delta_para' \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=3 \
    --print_freq=10 \
    --badnum=2 \
    --badrange=0.1 \
    2>&1 | tee $dir'/'$trial'_a=1_b=1_localep=50.txt'

done
