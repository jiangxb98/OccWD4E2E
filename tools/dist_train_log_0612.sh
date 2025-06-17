#!/bin/bash

# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/jgf/OccWD4E2E"
cd /cache/jgf/OccWD4E2E

# 创建日志文件夹
# 这个路径需要mkdir在云道通过url输出的路径下(echo $OUTPUT_URL/), 换到绿区后这个地址就变了
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/home/ma-user/modelarts/outputs/train_url_0

# modify
# python tools/utils/modify_pth.py --pth_path work_dirs/action_condition_MMO_MSO_plan_wo_gt_from_scratch/epoch_24.pth --key plan_head


# 对比6e
bash ./tools/dist_train.sh projects/configs/my/plan_rl_ft_query_1_im.py 8 \
    --work-dir work_dirs/plan_im_1_6e \
    --load-from work_dirs/action_condition_MMO_MSO_plan_wo_gt_from_scratch/epoch_24_modified.pth \
    --cfg-options runner.max_epochs=6

# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_im_freeze_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{
    bash ./tools/dist_test.sh projects/configs/my/plan_rl_ft_query_1_im.py \
        work_dirs/plan_im_1_6e/latest.pth 8
} 2>&1 | tee ${log_file}



# 12e
bash ./tools/dist_train.sh projects/configs/my/plan_rl_ft_query_1_im.py 8 \
    --work-dir work_dirs/plan_im_1_12e \
    --load-from work_dirs/action_condition_MMO_MSO_plan_wo_gt_from_scratch/epoch_24_modified.pth \
    --cfg-options runner.max_epochs=12

# 评估并记录日志
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/test_for_plan_im_freeze_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{
    bash ./tools/dist_test.sh projects/configs/my/plan_rl_ft_query_1_im.py \
        work_dirs/plan_im_1_12e/latest.pth 8
} 2>&1 | tee ${log_file}


# 占卡
echo "Grabbing GPUs..."
cd /cache
bash alive_8.sh
