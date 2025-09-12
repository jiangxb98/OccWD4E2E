# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/jgf/OccWD4E2E"
cd /cache/jgf/OccWD4E2E

# 创建日志文件夹
# 这个路径需要mkdir在云道通过url输出的路径下(echo $OUTPUT_URL/), 换到绿区后这个地址就变了
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/home/ma-user/modelarts/outputs/train_url_0


bash ./tools/dist_train.sh projects/configs/my/plan_rl_ft_query_1_im_sim_num5_same.py 8 \
    --work-dir work_dirs/plan_im_sim_num5_same_12e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=12

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/plan_im_sim_num5_same_12e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/plan_rl_ft_query_1_im_sim_num5_same.py \
        work_dirs/plan_im_sim_num5_same_12e/epoch_12.pth 8

} 2>&1 | tee ${log_file}





bash ./tools/dist_train.sh projects/configs/my/plan_rl_ft_query_1_im_sim_num5_detach_sim_same.py 8 \
    --work-dir work_dirs/plan_im_sim_num5_detach_sim_same_12e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=12

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/plan_im_sim_num5_detach_sim_same_12e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/plan_rl_ft_query_1_im_sim_num5_detach_sim_same.py \
        work_dirs/plan_im_sim_num5_detach_sim_same_12e/epoch_12.pth 8

} 2>&1 | tee ${log_file}



bash ./tools/dist_train.sh projects/configs/my/plan_rl_ft_query_1_im_sim_num5_detach_sim_deep.py 8 \
    --work-dir work_dirs/plan_im_sim_num5_detach_sim_deep_12e \
    --load-from pretrained/epoch_12_two_backbone_im.pth \
    --cfg-options runner.max_epochs=12

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/plan_im_sim_num5_detach_sim_deep_12e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    bash ./tools/dist_test.sh projects/configs/my/plan_rl_ft_query_1_im_sim_num5_detach_sim_deep.py \
        work_dirs/plan_im_sim_num5_detach_sim_deep_12e/epoch_12.pth 8

} 2>&1 | tee ${log_file}

#########################################################################################################
#                                                  占卡                                                 #
echo "Grabbing GPUs..."
cd /cache
bash alive_8.sh
