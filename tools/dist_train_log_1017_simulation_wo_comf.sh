# 终止之前的进程
echo "Killing previous processes..." 
pkill -f alive

echo "Changing directory to /cache/jgf/OccWD4E2E"
cd /cache/jgf/OccWD4E2E

# 创建日志文件夹
# 这个路径需要mkdir在云道通过url输出的路径下(echo $OUTPUT_URL/), 换到绿区后这个地址就变了
# 已经创建好了 /cache/guangfeng/wd4e2e/output_logs_jgf
LOG_DIR=/home/ma-user/modelarts/outputs/train_url_0

# wo comf
bash ./tools/dist_train.sh projects/configs/my/im_sim_ablation/imitation_simulation_ablation_wo_comf.py 8 \
    --work-dir work_dirs/im_sim_ablation_wo_comf_12e \
    --load-from work_dirs/action_condition_MMO_MSO_plan_wo_gt_from_scratch/epoch_24_modified.pth \
    --cfg-options runner.max_epochs=12

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${LOG_DIR}/im_sim_ablation_wo_comf_12e_${current_time}.log"
echo "Running evaluation for fine-tuning model, logging to: ${log_file}"
{

    # im+sim
    bash ./tools/dist_test.sh projects/configs/my/im_sim_ablation/imitation_simulation_ablation_wo_comf.py work_dirs/im_sim_ablation_wo_comf_12e/epoch_12.pth 8
    # wo reward 可以共用
    bash ./tools/dist_test.sh projects/configs/my/plan_ft_freeze.py work_dirs/im_sim_ablation_wo_comf_12e/epoch_12.pth 8
    # w imitation 可以共用
    bash ./tools/dist_test.sh projects/configs/my/im_sim_ablation/imitation_simulation_inference_only_with_im.py work_dirs/im_sim_ablation_wo_comf_12e/epoch_12.pth 8

} 2>&1 | tee ${log_file}
