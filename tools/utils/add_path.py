######################################################################
# 这个脚本将一个模型的指定参数添加到另一个模型中
######################################################################

import torch
import copy

# 加载模型
path = 'work_dirs/action_condition_MMO_MSO_plan_wo_gt_from_scratch/epoch_24.pth'
path_source = 'pretrained/r101_dcn_fcos3d_pretrain.pth'  # 模型，用于添加参数
model = torch.load(path)
model_source = torch.load(path_source)

# 指定添加的参数
add_keys = ['reward_model']

# 检查模型结构，找到img_backbone和img_neck参数
if isinstance(model, dict):
    # 如果是状态字典格式
    state_dict = model['state_dict']
    state_dict_source = model_source['state_dict']
    
    # 添加参数
    for key in add_keys:
        # 在源模型中查找包含指定关键字的参数
        for source_key, source_value in state_dict_source.items():
            if key in source_key:
                # 检查目标模型中是否已存在该参数
                if source_key not in state_dict:
                    state_dict[source_key] = copy.deepcopy(source_value)
                    print(f"添加参数: {source_key}")
                else:
                    print(f"参数已存在，跳过: {source_key}")
    
    # 更新模型的state_dict
    model['state_dict'] = state_dict
    
else:
    # 如果模型直接是状态字典格式
    state_dict = model
    state_dict_source = model_source
    
    # 添加参数
    for key in add_keys:
        # 在源模型中查找包含指定关键字的参数
        for source_key, source_value in state_dict_source.items():
            if key in source_key:
                # 检查目标模型中是否已存在该参数
                if source_key not in state_dict:
                    state_dict[source_key] = copy.deepcopy(source_value)
                    print(f"添加参数: {source_key}")
                else:
                    print(f"参数已存在，跳过: {source_key}")

# 保存模型
new_path = path.replace('.pth', '_add_model.pth')
torch.save(model, new_path)
print(f"模型已保存为{new_path}")
