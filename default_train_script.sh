CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py train \
    -o models/model.pt \
    --dataset_path big_data \
    --pretrain_model ibm-granite/granite-3.1-1b-a400m-base \
    --learning_rate 0.000003 \
    --batch_size 16 \
    --accumulation_steps 2 \
    --n_epochs 10 \
    --report_to wandb