# if use fine-grained, do not use generate_data.sh
CONFIG=projects/configs/inflated/action_condition_GMO.py
GPU_NUM=8   # 8 GPUs(Processes) consume ~20 hours

# Generate training data
./tools/dist_train.sh ${CONFIG} ${GPU_NUM}  --work-dir work_dirs/action_condition_GMO_generate_data

# Generate validation data
CKPT=work_dirs/action_condition_GMO_generate_data/epoch_1.pth
./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}



# for train
# CONFIG=projects\configs\fine_grained\action_condition_MMO_MSO.py
# GPU_NUM=8

# ./tools/dist_train.sh ${CONFIG} ${GPU_NUM}


# for val
# CONFIG=projects\configs\fine_grained\action_condition_MMO_MSO.py
# CKPT=work_dirs/action_condition_MMO_MSO/epoch_24.pth
# GPU_NUM=8

# ./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}


# 占卡
bash modelarts/user-job-dir/code/valley/train/grab_gpu/alive_8.sh