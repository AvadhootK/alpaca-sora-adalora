export WANDB_DISABLED=true
python train.py \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir Llama2-7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \

# finetune script for alpaca-lora
python finetune.py \
--base_model 'NousResearch/Llama-2-7b-hf' \
--data_path './alpaca_data.json' \
--output_dir './lora-alpaca' \
--batch_size 16 \
--micro_batch_size 2 \
--num_epochs 3 \
--learning_rate 1e-4 \
--cutoff_len 512 \
--val_set_size 2000 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules '[q_proj,v_proj,k_proj]' \
--train_on_inputs \
--group_by_length

# sora
python train_sora.py \
--base_model 'NousResearch/Llama-2-7b-hf' \
--data_path './alpaca_data.json' \
--output_dir './sora-alpaca' \
--batch_size 16 \
--micro_batch_size 2 \
--num_epochs 3 \
--learning_rate 1e-4 \
--cutoff_len 512 \
--val_set_size 2000 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules '[q_proj,v_proj,k_proj]' \
--train_on_inputs \
--group_by_length 
