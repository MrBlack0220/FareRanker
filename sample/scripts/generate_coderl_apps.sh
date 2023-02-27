model_path=../models/codet5_finetuned_codeRL
tokenizer_path=../models/codet5_tokenizer/
test_path=/root/APPS/train

start=0
end=5000
num_seqs_per_iter=25
num_seqs=100
temp=0.6

output_path=outputs/coderl-train

CUDA_VISIBLE_DEVICES=0 python coderl_generate_apps.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp \
