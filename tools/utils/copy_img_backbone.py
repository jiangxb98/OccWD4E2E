######################################################################
# 这个脚本用于将一个pretrained模型的img_backbone和img_neck参数复制到另一个模型中
######################################################################

import torch
import copy

# 加载模型
path = 'work_dirs/action_condition_MMO_MSO_plan_wo_gt_from_scratch/epoch_24.pth'
path_2 = 'pretrained/r101_dcn_fcos3d_pretrain.pth'  # 预训练模型
model = torch.load(path)
model_2 = torch.load(path_2)

# 检查模型结构，找到img_backbone和img_neck参数
if isinstance(model, dict):
    # 如果是状态字典格式
    state_dict = model['state_dict']
    state_dict_2 = model_2['state_dict']
    
    # 复制img_backbone到img_backbone_v2
    img_backbone_keys = [key for key in state_dict.keys() if key.startswith('img_backbone')]
    for key in img_backbone_keys:
        if not key.startswith('img_backbone_v2'):  # 避免重复复制
            new_key = key.replace('img_backbone', 'img_backbone_v2')
            state_dict[new_key] = state_dict_2[key].clone()
    
    # 复制img_neck到img_neck_v2
    img_neck_keys = [key for key in state_dict.keys() if key.startswith('img_neck')]
    for key in img_neck_keys:
        if not key.startswith('img_neck_v2'):  # 避免重复复制
            new_key = key.replace('img_neck', 'img_neck_v2')
            state_dict[new_key] = state_dict_2[key].clone()
    
    print(f"已复制 {len(img_backbone_keys)} 个img_backbone参数到img_backbone_v2")
    print(f"已复制 {len(img_neck_keys)} 个img_neck参数到img_neck_v2")
    

# 保存模型
new_path = path.replace('.pth', '_two_backbone.pth')
torch.save(model, new_path)
print(f"模型已保存为{new_path}")
