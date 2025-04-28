#!/bin/bash

# 创建日志文件夹
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf

# 获取当前时间作为日志文件名
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="/cache/guangfeng/wd4e2e/output_logs_jgf/train_${current_time}.log"

# 写入日志头部信息
echo "=== Training Log ===" | tee ${log_file}
echo "Start Time: $(date)" | tee -a ${log_file}
echo "===================" | tee -a ${log_file}
echo "" | tee -a ${log_file}

# 重定向所有后续命令的输出到终端和日志文件
exec &> >(tee -a "$log_file")

# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory..."
cd /cache/guangfeng/wd4e2e/OccWD4E2E

# for train
echo "Starting training..."
CONFIG=projects/configs/fine_grained/action_condition_MMO_MSO.py
GPU_NUM=8

echo "Running with config: ${CONFIG} and ${GPU_NUM} GPUs"
bash ./tools/dist_train.sh ${CONFIG} ${GPU_NUM} --work-dir work_dirs/action_condition_MMO_MSO

# for val
echo "Starting validation..."
CONFIG=projects/configs/fine_grained/action_condition_MMO_MSO.py
CKPT=work_dirs/action_condition_MMO_MSO/epoch_24.pth
GPU_NUM=8

echo "Running validation with checkpoint: ${CKPT}"
bash ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}

# 占卡
echo "Grabbing GPUs..."
cd ~/modelarts/user-job-dir/code/valley/train/grab_gpu
bash alive_8.sh

# 写入日志结束信息
echo "" | tee -a ${log_file}
echo "=== Training Completed ===" | tee -a ${log_file}
echo "End Time: $(date)" | tee -a ${log_file}
echo "===================" | tee -a ${log_file}