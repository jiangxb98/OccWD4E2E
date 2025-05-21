#!/bin/bash

# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/guangfeng/wd4e2e/OccWD4E2E"
cd /cache/guangfeng/wd4e2e/OccWD4E2E

# 创建日志文件夹
# 这个路径需要mkdir在云道通过url输出的路径下(echo $OUTPUT_URL/)，然后软链接到下面这个地址(/cache/guangfeng/wd4e2e/output_logs_jgf)
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/cache/guangfeng/wd4e2e/output_logs_jgf

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
    bash ./tools/dist_test.sh projects/configs/fine_grained/action_condition_MMO_MSO_plan_ft_unfreeze.py \
        work_dirs/action_condition_MMO_MSO_plan_ft_unfreeze/latest.pth 8
} 2>&1 | tee ${log_file}


# 占卡
echo "Grabbing GPUs..."
cd ~/modelarts/user-job-dir/code/valley/train/grab_gpu
bash alive_8.sh

