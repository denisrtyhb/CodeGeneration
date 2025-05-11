CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py train \
    --dataset_path huge_dataset \
    --pretrain_model ibm-granite/granite-3.1-1b-a400m-base \
    --learning_rate 0.00001 \
    --batch_size 16 \
    --accumulation_steps 8 \
    --n_epochs 40 \
    --report_to wandb \
    --output_path models_hugest/model.pt | tee exp6.log