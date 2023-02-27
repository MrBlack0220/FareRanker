code_path=../sample/outputs/coderl-train
output_path=outputs/coderl-train
test_path=/root/APPS/train
example_tests = 0 # 0: run hidden unit tests; 1: run example unit tests 
start=0
end=5000
threads=1

if [ ! -d $output_path ] 
then
    echo "Directory DOES NOT exists." 
    mkdir $output_path
fi

index=0
for (( i=$start;i<$end;i++ )) ; do 
    echo 'testing sample index #' ${i}
    ((index++))   
    (
    python test_one_solution.py \
        --code_path ${code_path} \
        --output_path ${output_path} \
        --test_path $test_path \
        --example_tests $example_tests \
        --i $i 
    ) &        
    if (( $index % $threads == 0 )); then wait; fi 
done 

wait 
