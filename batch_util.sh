# This is utility script to run inference on a dataset using 26B model, with multiple SLURM jobs.
# This script is used to run inference on a dataset that has been split using split.py script.
# Be careful with each job's GPU memory usage, as the model is large and may not fit in a single GPU. (you need 2x24GB GPUs, so set 2 if you have rtx3090 nodes)

splits=4
partition=gpux
qos=qos
gpus=1
cpus_per_gpu=16
python=~/InternVL/internvl/bin/python
infer_script=~/InternVL/tools/infer_26B.py
DATENUM=$(date +%Y%m%d%H%M%S) # this is for saving batch scripts, maybe check periodically to rerun preemptioned jobs

infer_dir=/result/dataset_cropped
index_dir=$infer_dir/index
batch_script_save_dir=~/InternVL/tools/batch_scripts/$DATENUM
output_dir=/result/cropped_0_result

mkdir -p $output_dir

split_script=~/InternVL/tools/split.py
# split is recommended before running this script

python $split_script --dataset_dir $infer_dir --split_count $splits --skip_validate

mkdir -p $batch_script_save_dir

for i in $(seq 0 $((splits-1))); do
    echo "srun --partition=$partition --time=72:0:0 --nodes=1 --cpus-per-gpu=$cpus_per_gpu --qos=$qos --gres=gpu:$gpus $python $infer_script --infer_file $index_dir/sanitized_indices_${i}.txt --infer_dir $infer_dir --skip_existing"
    echo "#!/bin/bash" > $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --job-name=vl26B_${i}" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --output=O-%x.%j" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --error=E-%x.%j" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --partition=$partition" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --nodes=1" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --gres=gpu:$gpus" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --time=72:00:00" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --cpus-per-gpu=$cpus_per_gpu" >> $batch_script_save_dir/batch_${i}.sh
    echo "#SBATCH --qos=$qos" >> $batch_script_save_dir/batch_${i}.sh
    echo "" >> $batch_script_save_dir/batch_${i}.sh
    echo "$python $infer_script --infer_file $index_dir/sanitized_indices_${i}.txt --infer_dir $infer_dir --skip_existing --output_dir $output_dir" >> $batch_script_save_dir/batch_${i}.sh
    chmod +x $batch_script_save_dir/batch_${i}.sh
    sbatch $batch_script_save_dir/batch_${i}.sh
done
