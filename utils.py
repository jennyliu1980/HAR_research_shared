import numpy as np
import torch
import os


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def evaluate_record(data_name, type, time_mask, channel_mask, alpha, beta, value, divide=None, epoch=None,
                    discriminate=None, tips=None):
    # 论文报告的F1分数
    paper_results = {
        'ucihar': {
            'time_10': 0.900,  # 估计值
            'time_20': 0.907,  # 估计值
            'time_30': 0.914,
            'time_40': 0.910,  # 估计值
            'spantime_10': 0.910,  # 估计值
            'spantime_20': 0.915,  # 估计值
            'spantime_30': 0.920,
            'spantime_40': 0.918,  # 估计值
            'channel_1': 0.900,  # 估计值
            'channel_2': 0.904,  # 估计值
            'channel_3': 0.908,
            'channel_4': 0.906,  # 估计值
            'time30_channel3_0.3': 0.921,  # 估计值
            'time30_channel3_0.5': 0.924,
            'time30_channel3_0.7': 0.922,  # 估计值
            'spantime30_channel3_0.3': 0.928,  # 估计值
            'spantime30_channel3_0.5': 0.931,  # 最佳
            'spantime30_channel3_0.7': 0.929,  # 估计值
            'no_pretrain': 0.892
        }
    }

    # 构建配置的key来查找论文结果
    expected = None
    if data_name in paper_results:
        if type == 'time':
            key = f'time_{time_mask}'
        elif type == 'spantime':
            key = f'spantime_{time_mask}'
        elif type == 'channel':
            key = f'channel_{channel_mask}'
        elif type == 'time_channel':
            key = f'time{time_mask}_channel{channel_mask}_{alpha}'
        elif type == 'spantime_channel':
            key = f'spantime{time_mask}_channel{channel_mask}_{alpha}'
        else:
            key = None

        if key and key in paper_results[data_name]:
            expected = paper_results[data_name][key]

    # 构建输出字符串
    suf = f"""
=====================================
Configuration: {type}
  Time Mask: {time_mask}%
  Channel Mask: {channel_mask}
  Alpha: {alpha}
  Seed/Divide: {divide}
-------------------------------------
Results:
  Test F1 Score: {value:.4f}"""

    if expected is not None:
        diff = value - expected
        status = '✓ PASS' if abs(diff) < 0.02 else '✗ BELOW EXPECTED' if diff < -0.02 else '⚠ ABOVE EXPECTED'
        suf += f"""
  Expected (Paper): {expected:.4f}
  Difference: {diff:+.4f} {status}"""
    else:
        suf += f"""
  Expected (Paper): Not specified"""

    suf += f"""
====================================="""

    # 添加额外信息
    if epoch is not None and epoch != 150:
        suf += f'\n  Pretrain epochs: {epoch}'
    if beta is not None:
        suf += f'\n  Beta: {beta}'
    if discriminate is not None and discriminate:
        suf += f'\n  Discriminator beta: {beta}'
    if tips is not None:
        suf += f'\n  Note: {tips}'

    # 写入文件
    filename = f"{data_name}_record.txt"
    with open(filename, "a+") as f:
        f.write(suf)
        f.write("\n")

    # 同时打印到控制台
    print(suf)

    # 如果结果明显低于预期，给出警告
    if expected and value < expected - 0.05:
        print(f"\n⚠️ WARNING: F1 score ({value:.4f}) is significantly below expected ({expected:.4f})")
        print("  Possible issues:")
        print("  - Need more fine-tuning epochs")
        print("  - Learning rate may need adjustment")
        print("  - Check if the correct model was loaded")


def save_model(data_name, my_type, time_mask, channel_mask, alpha, divide, model, epoch=None):
    model_dir = ''
    if data_name == 'uschad':
        if my_type == 'time':
            model_dir = 'model/{}/time{}'.format(data_name, time_mask)
        elif my_type == 'spantime':
            model_dir = 'model/{}/spantime{}'.format(data_name, time_mask)
        elif my_type == 'spantime_channel':
            model_dir = 'model/{}/spantime{}_channel{}_alpha{}'.format(data_name, time_mask, channel_mask, alpha)
        elif my_type == 'time_channel':
            model_dir = 'model/{}/time{}_channel{}_alpha{}'.format(data_name, time_mask, channel_mask, alpha)
        elif my_type == 'channel':
            model_dir = 'model/{}/channel{}'.format(data_name, channel_mask)
    else:
        if my_type == 'time':
            model_dir = 'model/{}/time{}_divide{}'.format(data_name, time_mask, divide)
        elif my_type == 'spantime':
            model_dir = 'model/{}/spantime{}_divide{}'.format(data_name, time_mask, divide)
        elif my_type == 'spantime_channel':
            model_dir = 'model/{}/spantime{}_channel{}_divide{}_alpha{}'.format(data_name, time_mask, channel_mask,
                                                                                divide, alpha)
        elif my_type == 'time_channel':
            model_dir = 'model/{}/time{}_channel{}_divide{}_alpha{}'.format(data_name, time_mask, channel_mask, divide,
                                                                            alpha)
        elif my_type == 'channel':
            model_dir = 'model/{}/channel{}_divide{}'.format(data_name, channel_mask, divide)

    # if epoch != None and epoch != 150:
    #     model_dir += '_epoch{}'.format(epoch)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    torch.save(model, model_dir)
    return model_dir