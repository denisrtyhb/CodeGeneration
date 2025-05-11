CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python codegen/framework/main.py \
    --dataset_path framework_dataset \
    --model_name ibm-granite/granite-3.1-1b-a400m-base \
    --model_path "" \
    --output_path results/not_trained_hugest