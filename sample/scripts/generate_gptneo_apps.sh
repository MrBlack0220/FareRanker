model_path=../models/01-19-2023__14:58:34/final_checkpoint
tokenizer_path=EleutherAI/gpt-neo-125M
test_path=/root/APPS/train

start=0
end=5000
num_seqs_per_iter=25
num_seqs=100
temp=0.9

output_path=outputs/gpt-neo-125-train

CUDA_VISIBLE_DEVICES=0 python gptneo125_generate_apps.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp \
