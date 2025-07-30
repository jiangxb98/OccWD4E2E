# load 一个模型，然后复制img_backbone到img_backbone_v2，这个pth文件拥有两套img_backbone的参数

import torch
import copy

# 加载模型
model = torch.load('model.pth')

# 检查模型结构，找到img_backbone和img_neck参数
if isinstance(model, dict):
    # 如果是状态字典格式
    state_dict = model
    
    # 复制img_backbone到img_backbone_v2
    img_backbone_keys = [key for key in state_dict.keys() if key.startswith('img_backbone')]
    for key in img_backbone_keys:
        if not key.startswith('img_backbone_v2'):  # 避免重复复制
            new_key = key.replace('img_backbone', 'img_backbone_v2')
            state_dict[new_key] = state_dict[key].clone()
    
    # 复制img_neck到img_neck_v2
    img_neck_keys = [key for key in state_dict.keys() if key.startswith('img_neck')]
    for key in img_neck_keys:
        if not key.startswith('img_neck_v2'):  # 避免重复复制
            new_key = key.replace('img_neck', 'img_neck_v2')
            state_dict[new_key] = state_dict[key].clone()
    
    print(f"已复制 {len(img_backbone_keys)} 个img_backbone参数到img_backbone_v2")
    print(f"已复制 {len(img_neck_keys)} 个img_neck参数到img_neck_v2")
    
else:
    # 如果是完整的模型对象
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        
        # 复制img_backbone到img_backbone_v2
        img_backbone_keys = [key for key in state_dict.keys() if key.startswith('img_backbone')]
        for key in img_backbone_keys:
            if not key.startswith('img_backbone_v2'):  # 避免重复复制
                new_key = key.replace('img_backbone', 'img_backbone_v2')
                state_dict[new_key] = state_dict[key].clone()
        
        # 复制img_neck到img_neck_v2
        img_neck_keys = [key for key in state_dict.keys() if key.startswith('img_neck')]
        for key in img_neck_keys:
            if not key.startswith('img_neck_v2'):  # 避免重复复制
                new_key = key.replace('img_neck', 'img_neck_v2')
                state_dict[new_key] = state_dict[key].clone()
        
        # 更新模型的状态字典
        model.load_state_dict(state_dict)
        print(f"已复制 {len(img_backbone_keys)} 个img_backbone参数到img_backbone_v2")
        print(f"已复制 {len(img_neck_keys)} 个img_neck参数到img_neck_v2")
    else:
        print("模型格式不支持，请检查模型结构")

# 保存模型
torch.save(model, 'model_v2.pth')
print("模型已保存为 model_v2.pth")


