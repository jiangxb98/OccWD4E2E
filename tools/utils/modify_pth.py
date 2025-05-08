# 写一个python脚本读取一个pth文件，然后将指定key的值删掉，然后保存到新的pth文件中

import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, default="model.pth")
    # key is the list for the keys to be deleted
    parser.add_argument("--key", nargs="+", type=str, default=['plan_head'])

    args = parser.parse_args()

    # 读取pth文件
    state_dict = torch.load(args.pth_path)

    # 删除指定key的值
    for key in args.key:
        del state_dict[key]

    # 保存到新的pth文件中
    output_path = args.pth_path.replace(".pth", "_modified.pth")
    torch.save(state_dict, output_path)
    print(f"Modified pth file saved to {output_path}")
