  python -m torch.distributed.launch --nproc_per_node 4 --master_port='29500' --use_env run_ranker.py \
    --train_data_path ../generate_final_dataset/dataset/coderl/train/genaration_8_translation_8_tokenizedata \
    --dev_data_path ../generate_final_dataset/dataset/coderl/eval/translation_2_tokenizedata.pkl \
    --test_data_path ../generate_final_dataset/dataset/coderl/test/translation_2_tokenizedata.pkl \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 100\
    --gradient_accumulation_steps 64 \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.2 \
    --evaluation_strategy steps --eval_steps 30 \
    --logging_strategy steps --logging_steps  5\
    --logging_first_step True \
    --save_strategy steps --save_steps 30 --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_ranked_pass@1 --greater_is_better True \
    --model_name_or_path microsoft/unixcoder-base \
    --output_dir outputs \
    --step1_max_num 8 \
    --step2_max_num 8 \
    --loss_type contrastive \
    --max_source_length 600 --max_candidate_length 512 \
    --disable_tqdm False \
    --temperature 0.5\


