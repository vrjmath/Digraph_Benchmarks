#export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader)
export CUDA_VISIBLE_DEVICES=0


for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --undirected --wandb --GNN="GIN" --num_layers=1 --log_file_name="downstream_log_unGIN.txt" --seed=$seed &
done


wait
echo "Experiments completed"

