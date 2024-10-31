<<'COMMENT'

export CUDA_VISIBLE_DEVICES=0
(
for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --wandb --GNN="GIN" --num_layers=1 --log_file_name="finetune_edge_prediction_GIN1k.txt" --pretrained_model="pretrained_models/edge_predictionGIN1k.pth" --fine_tuning --seed=$seed &
done
) &

export CUDA_VISIBLE_DEVICES=1
(
for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --bidirectional --wandb --GNN="GIN" --num_layers=1 --log_file_name="finetune_edge_prediction_biGIN1k.txt" --pretrained_model="pretrained_models/edge_predictionbiGIN1k.pth" --fine_tuning --seed=$seed &
done
) &

export CUDA_VISIBLE_DEVICES=5
(
for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --undirected --wandb --GNN="GIN" --num_layers=1 --log_file_name="finetune_edge_prediction_unGIN1k.txt" --pretrained_model="pretrained_models/edge_predictionunGIN1k.pth" --fine_tuning --seed=$seed &
done
) &
COMMENT

export CUDA_VISIBLE_DEVICES=0
(
for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --wandb --GNN="SAGE" --num_layers=2 --log_file_name="freeze_finetune_directed_prediction_SAGE1k.txt" --pretrained_model="pretrained_models/directed_predictionSAGE1k.pth" --fine_tuning --seed=$seed &
done
) &

export CUDA_VISIBLE_DEVICES=1
(
for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --bidirectional --wandb --GNN="SAGE" --num_layers=2 --log_file_name="freeze_finetune_directed_prediction_biSAGE1k.txt" --pretrained_model="pretrained_models/directed_predictionbiSAGE1k.pth" --fine_tuning --seed=$seed &
done
) &

export CUDA_VISIBLE_DEVICES=5
(
for seed in 42 27 6 81 89 17 99 23 18 68; do
    nohup python -u train.py --num_threads=2 --batch_size=2 --epochs=1000 --patience=20 --undirected --wandb --GNN="SAGE" --num_layers=2 --log_file_name="freeze_finetune_directed_prediction_unSAGE1k.txt" --pretrained_model="pretrained_models/directed_predictionunSAGE1k.pth" --fine_tuning --seed=$seed &
done
) &


wait
echo "Experiments completed"

