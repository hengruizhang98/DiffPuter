for name in default shoppers
do
    for idx in 0 1 2 3
    do
        python main.py --dataname $name --split_idx $idx
    done
done


# nohup bash run.sh > log.txt 2>&1 &