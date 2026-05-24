
generate_run_id() {
    local model_version=$1
    local table_id=$2
    local regression_targets=$3
    local covariates=$4
    local cost_function_regression=$5
    local weight_stage=$6
    local weight_regression_targets=$7
    local weight_classification=$8
    local n_training_samples=$9
    local learning_rate=${10}
    local lr_gamma=${11}
    local regression_pre_final_dim=${12}
    local max_epochs=${13}
    local fold_id=${14}

    # Format the targets and weights as lists with underscores for spaces, wrapped in single quotes
    local formatted_regression_targets="['$(echo ${regression_targets} | sed "s/ /'_'/g")']"
    local formatted_covariates="['$(echo ${covariates} | sed "s/ /'_'/g")']"
    local formatted_weight_regression_targets="[$(echo ${weight_regression_targets} | sed "s/ /_/g")]"

    formatted_learning_rate=$(printf "%.4f" "$learning_rate")
    formatted_lr_gamma=$(printf "%.4f" "$lr_gamma")

    # Create the run_id
    local run_id="${model_version}_${table_id}_${formatted_regression_targets}_${formatted_covariates}_${cost_function_regression}_wSS${weight_stage}_wRe${formatted_weight_regression_targets}_wCl${weight_classification}_ntr${n_training_samples}_lr${formatted_learning_rate}-${formatted_lr_gamma}_d${regression_pre_final_dim}_ep${max_epochs}_f${fold_id}"

    echo ${run_id}
}

gpu_id=0

### MODEL AND DATA CONFIGURATION:
model_version="v2_3"
model_name="MaxxVit_oracle_${model_version}"
fs_time=1
config="wlt_${fs_time}hz_100f"
# table_id="master_mp3_${config}"
table_id="df_master_sample"

path_mastertable="${table_id}.csv"

for fold_id in 0; do #  1 2 3 4; do
    echo -e "\n\n\e[33mSTART FOLD ${fold_id}\e[0m"

    covariates="age sex"
    cost_function_regression="huber"
    regression_pre_final_dim=1024

    ### PIPELINE CONFIGURATION:
    do_base_training=True

    ### 1. Train base model:

    set_base_params() {
        global_regression_targets="cog-main age_z self"
        global_weight_regression_targets="0.8 0.05 0.15"
        global_classification_targets="dx"
        global_weight_classification=0.3
        global_weight_stage=0.04

        global_n_training_samples=5
        global_min_epochs=1 # 9
        global_max_epochs=1 # 9

        global_learning_rate=1e-4
        global_lr_gamma=2e-4 # weight decay in AdamW
    }

    set_base_params
    regression_targets=$global_regression_targets
    weight_regression_targets=$global_weight_regression_targets
    classification_targets=$global_classification_targets
    weight_classification=$global_weight_classification
    weight_stage=$global_weight_stage
    n_training_samples=$global_n_training_samples
    min_epochs=$global_min_epochs
    max_epochs=$global_max_epochs
    learning_rate=$global_learning_rate
    lr_gamma=$global_lr_gamma

    print_main_info=False

    if [ ${do_base_training} = True ]; then
        python main_train_model.py ${gpu_id} ${path_mastertable} ${model_name} --fold_id ${fold_id} --regression_targets ${regression_targets} --weight_regression_targets ${weight_regression_targets} --classification_targets ${classification_targets} --weight_classification ${weight_classification} --weight_stage ${weight_stage} --covariates ${covariates} --cost_function_regression ${cost_function_regression} --learning_rate ${learning_rate} --lr_gamma ${lr_gamma} --min_epochs ${min_epochs} --max_epochs ${max_epochs} --n_training_samples ${n_training_samples} --regression_pre_final_dim ${regression_pre_final_dim} --fs_time ${fs_time} --print_main_info ${print_main_info} 
    fi
    echo -e "\e[33mEND FOLD ${fold_id}\e[0m\n\n"
done
