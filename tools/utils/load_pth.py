import torch



def load_pth(pth_path):
    state_dict = torch.load(pth_path)
    return state_dict




if __name__ == "__main__":
    pth_path = "pth_path"
    state_dict = load_pth(pth_path)
    
    pth_path_2 = "pth_path_2"
    state_dict_2 = load_pth(pth_path_2)

    # 对比指定key的参数是否一致

    key_list = ["plan_head", "future_pred_head"]
   
    
    for k, v in state_dict.items():
        for key in key_list:
            if key in k:
                print(k)


