#!/bin/bash

# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/guangfeng/wd4e2e/OccWD4E2E"
cd /cache/guangfeng/wd4e2e/OccWD4E2E

# 创建日志文件夹
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/cache/guangfeng/wd4e2e/output_logs_jgf

# 2.2 微调6epoch，解冻其他参数
# 训练
bash ./tools/dist_train.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_ft_unfreeze.py 8 \
    --work-dir work_dirs/action_condition_MMO_MSO_plan_ft_unfreeze \
    --load-from /cache/guangfeng/wd4e2e/OccWD4E2E/work_dirs/action_condition_MMO_MSO/epoch_24.pth \
    --cfg-options runner.max_epochs=6

# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_ft_unfreeze_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{
    echo "=== Evaluation Log ==="
    echo "Start Time: $(date)"
    echo "Model: action_condition_MMO_MSO_plan_ft_unfreeze"
    echo "Checkpoint: work_dirs/action_condition_MMO_MSO_plan_ft_unfreeze/latest.pth"
    echo "==================="
    echo ""
    
    bash ./tools/dist_test.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_ft_unfreeze.py \
        work_dirs/action_condition_MMO_MSO_plan_ft_unfreeze/latest.pth 8
    
    echo ""
    echo "=== Evaluation Completed ==="
    echo "End Time: $(date)"
    echo "==================="
} 2>&1 | tee ${log_file}



# 2.3 直接训24epoch
bash ./tools/dist_train.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan.py 8 \
    --work-dir work_dirs/action_condition_MMO_MSO_plan_24e

# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_24e_${current_time}.log"
echo "Running evaluation for 24-epoch model, logging to: ${log_file}"
{
    echo "=== Evaluation Log ==="
    echo "Start Time: $(date)"
    echo "Model: action_condition_MMO_MSO_plan_24e"
    echo "Checkpoint: work_dirs/action_condition_MMO_MSO_plan_24e/latest.pth"
    echo "==================="
    echo ""
    
    bash ./tools/dist_test.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan.py \
        work_dirs/action_condition_MMO_MSO_plan_24e/latest.pth 8
    
    echo ""
    echo "=== Evaluation Completed ==="
    echo "End Time: $(date)"
    echo "==================="
} 2>&1 | tee ${log_file}

# 2.4 训练30e， 这个之后再设置，先不用

# 3 使用Reward Model训练
# 3.1 监督最后一帧的结果
# 3.1.1 plan_query_nums = 1 for imitation
bash ./tools/dist_train.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_rl_ft_query_1_im.py 8 \
    --work-dir work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_1_im \
    --load-from /cache/guangfeng/wd4e2e/OccWD4E2E/work_dirs/action_condition_MMO_MSO/epoch_24.pth \
    --cfg-options runner.max_epochs=6
# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_rl_ft_query_1_im_${current_time}.log"
echo "Running evaluation for 24-epoch model, logging to: ${log_file}"
{
    echo "=== Evaluation Log ==="
    echo "Start Time: $(date)"
    echo "Model: action_condition_MMO_MSO_plan_rl_ft_query_1_im"
    echo "Checkpoint: work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_1_im/latest.pth"
    echo "==================="
    echo ""
    
    bash ./tools/dist_test.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_rl_ft_query_1_im.py \
        work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_1_im/latest.pth 8
    
    echo ""
    echo "=== Evaluation Completed ==="
    echo "End Time: $(date)"
    echo "==================="
} 2>&1 | tee ${log_file}


# 3.1.2 plan_query_nums = 1 for imitation and simulation
bash ./tools/dist_train.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_rl_ft_query_1_im_sim.py 8 \
    --work-dir work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_1_im_sim \
    --load-from /cache/guangfeng/wd4e2e/OccWD4E2E/work_dirs/action_condition_MMO_MSO/epoch_24.pth \
    --cfg-options runner.max_epochs=6
# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_rl_ft_query_1_im_sim_${current_time}.log"
echo "Running evaluation for 24-epoch model, logging to: ${log_file}"
{
    echo "=== Evaluation Log ==="
    echo "Start Time: $(date)"
    echo "Model: action_condition_MMO_MSO_plan_rl_ft_query_1_im_sim"
    echo "Checkpoint: work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_1_im_sim/latest.pth"
    echo "==================="
    echo ""
    
    bash ./tools/dist_test.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_rl_ft_query_1_im_sim.py \
        work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_1_im_sim/latest.pth 8
    
    echo ""
    echo "=== Evaluation Completed ==="
    echo "End Time: $(date)"
    echo "==================="
} 2>&1 | tee ${log_file}



# 3.1.3 plan_query_nums = k for imitation and simulation
bash ./tools/dist_train.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_rl_ft_query_k_im_sim.py 8 \
    --work-dir work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_k_im_sim \
    --load-from /cache/guangfeng/wd4e2e/OccWD4E2E/work_dirs/action_condition_MMO_MSO/epoch_24.pth \
    --cfg-options runner.max_epochs=6
# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_rl_ft_query_k_im_sim_${current_time}.log"
echo "Running evaluation for 24-epoch model, logging to: ${log_file}"
{
    echo "=== Evaluation Log ==="
    echo "Start Time: $(date)"
    echo "Model: action_condition_MMO_MSO_plan_rl_ft_query_k_im_sim"
    echo "Checkpoint: work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_k_im_sim/latest.pth"
    echo "==================="
    echo ""
    
    bash ./tools/dist_test.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_rl_ft_query_k_im_sim.py \
        work_dirs/action_condition_MMO_MSO_plan_rl_ft_query_k_im_sim/latest.pth 8
    
    echo ""
    echo "=== Evaluation Completed ==="
    echo "End Time: $(date)"
    echo "==================="
} 2>&1 | tee ${log_file}


# 占卡
echo "Grabbing GPUs..."
cd ~/modelarts/user-job-dir/code/valley/train/grab_gpu
bash alive_8.sh

