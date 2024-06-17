experiment_id=$(date +'%Y%m%d_%H%M%S')
log_dir="experiments/log/$experiment_id/splits/"
mkdir -p "$log_dir"


# Run Python script 5 times with different seeds
for ((seed = 42; seed <= 46; seed++)); do
    seed_log_dir="$log_dir/$seed/"
    mkdir -p "$seed_log_dir"
    
    python detectors/R2-D2/train/train.py --experiment_id $experiment_id --mw_folder data/features/original/GM19/mw/R2-D2/images --gw_folder data/features/original/GM19/gw/R2-D2/images --seed $seed --log_dir $seed_log_dir
done
