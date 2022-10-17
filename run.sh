#!bash
freeze=('direct_train' 'direct_freeze') # 'delta_tuning'
train_size=(0 2 4 8 16 32 64 128 177)
# data_path=("annotate_data/processed_data_char.xlsx")#"annotate_data/processed_data.xlsx" "annotate_data/processed_data_sen.xlsx" 
random_seed=(123 234 456 789 983)  
for j in {0..1}
do
    for k in {0..5}
    do
        for i in {0..9}
        do
            python finetune_model.py --prompt 1 --train_size ${train_size[i]} --freeze 'direct_train' --random_seed ${random_seed[k]} --dataset_type 'qqp' --direct_prompt 0 --dataset_name "annotate_data/processed_data_char.xlsx"
        done
    done
done

