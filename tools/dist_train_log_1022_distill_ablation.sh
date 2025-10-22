# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/jgf/OccWD4E2E"
cd /cache/jgf/OccWD4E2E

# 创建日志文件夹
# 这个路径需要mkdir在云道通过url输出的路径下(echo $OUTPUT_URL/), 换到绿区后这个地址就变了
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/home/ma-user/modelarts/outputs/train_url_0

# plan query
bash ./tools/dist_train.sh projects/configs/my/distill_query.py 8 \
    --work-dir work_dirs/distill_query_6e_start_predocc_3 \
    --load-from pretrained/epoch_24_two_backbone.pth \
    --cfg-options runner.max_epochs=6

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/distill_ablation_query_6e_start_predocc_3_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/distill_query.py \
        work_dirs/distill_query_6e_start_predocc_3/latest.pth 8

} 2>&1 | tee ${log_file}


# imitation reward
# with pred occ
# load的这个权重imitation是通过gt occ训练的
bash ./tools/dist_train.sh projects/configs/my/distill_im_reward.py 8 \
    --work-dir work_dirs/distill_im_reward_6e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=6

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/distill_ablation_im_reward_6e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/distill_im_reward.py \
        work_dirs/distill_im_reward_6e/latest.pth 8

} 2>&1 | tee ${log_file}


# plan query & imitation reward
# with pred occ with teacher traj
bash ./tools/dist_train.sh projects/configs/my/distill_im_query_reward.py 8 \
    --work-dir work_dirs/distill_im_query_reward_6e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=6

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/distill_ablation_im_query_reward_6e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/distill_im_query_reward.py \
        work_dirs/distill_im_query_reward_6e/latest.pth 8

} 2>&1 | tee ${log_file}


#########################################################################################################
#                                                  占卡                                                 #
echo "Grabbing GPUs..."
cd /cache
bash alive_8.sh
