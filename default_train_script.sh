CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py train \
    -o models/model.pt \
    --dataset_path big_data \
    --pretrain_model sentence-transformers/all-MiniLM-L6-v2 \
    --learning_rate 0.00001 \
    --batch_size 16 \
    --n_epochs 5 \
    --report_to wandb