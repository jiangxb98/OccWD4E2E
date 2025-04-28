pkill -f alive
cd /cache/guangfeng/wd4e2e/OccWD4E2E


# for train
CONFIG=projects/configs/fine_grained/action_condition_MMO_MSO.py
GPU_NUM=8

bash ./tools/dist_train.sh ${CONFIG} ${GPU_NUM}  --work-dir work_dirs/action_condition_MMO_MSO

# for val
CONFIG=projects/configs/fine_grained/action_condition_MMO_MSO.py
CKPT=work_dirs/action_condition_MMO_MSO/epoch_24.pth
GPU_NUM=8

bash ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}

# 占卡
cd ~/modelarts/user-job-dir/code/valley/train/grab_gpu
bash alive_8.sh