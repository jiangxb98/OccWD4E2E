# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/jgf/OccWD4E2E"
cd /cache/jgf/OccWD4E2E

# 创建日志文件夹
# 这个路径需要mkdir在云道通过url输出的路径下(echo $OUTPUT_URL/), 换到绿区后这个地址就变了
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/home/ma-user/modelarts/outputs/train_url_0


bash ./tools/dist_train.sh projects/configs/my/plan_transfer_distill_reward_detach_bev.py 8 \
    --work-dir work_dirs/plan_reward_distill_detach_bev_6e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=6

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/plan_reward_distill_detach_bev_6e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/plan_transfer_distill_reward_detach_bev.py \
        work_dirs/plan_reward_distill_detach_bev_6e/latest.pth 8

} 2>&1 | tee ${log_file}





bash ./tools/dist_train.sh projects/configs/my/plan_transfer_distill_reward_detach_bev_start_v1.py 8 \
    --work-dir work_dirs/plan_reward_distill_detach_bev_start_v1_6e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=6

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/plan_reward_distill_detach_bev_start_v1_6e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/plan_transfer_distill_reward_detach_bev_start_v1.py \
        work_dirs/plan_reward_distill_detach_bev_start_v1_6e/latest.pth 8

} 2>&1 | tee ${log_file}



bash ./tools/dist_train.sh projects/configs/my/plan_transfer_distill_reward_detach_bev_start_v2.py 8 \
    --work-dir work_dirs/plan_reward_distill_detach_bev_start_v2_12e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=6

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/plan_reward_distill_detach_bev_start_v2_12e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/plan_transfer_distill_reward_detach_bev_start_v2.py \
        work_dirs/plan_reward_distill_detach_bev_start_v2_12e/latest.pth 8

} 2>&1 | tee ${log_file}

#########################################################################################################
#                                                  占卡                                                 #
echo "Grabbing GPUs..."
cd /cache
bash alive_8.sh
