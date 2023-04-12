import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import shutil
import wave

WAV_MIN_LENGTH = 2    # wav文件的最短时长 / The minimum duration of wav files
SAMPLE_RATE = 1    # 抽取文件数量的百分比 / The percentage of files to be extracted
SAMPLE_MIN = 2    # 抽取的文件数量下限 / The lower limit of the number of files to be extracted
SAMPLE_MAX = 10    # 抽取的文件数量上限 / The upper limit of the number of files to be extracted


# 定义一个函数，用于检查wav文件的时长是否大于最短时长
def check_duration(wav_file):
    # 打开wav文件
    f = wave.open(wav_file, "rb")
    # 获取帧数和帧率
    frames = f.getnframes()
    rate = f.getframerate()
    # 计算时长（秒）
    duration = frames / float(rate)
    # 关闭文件
    f.close()
    # 返回时长是否大于最短时长的布尔值
    return duration > WAV_MIN_LENGTH

# 定义一个函数，用于从给定的目录中随机抽取一定比例的wav文件，并剪切到另一个目录中，保留数据结构
def split_data(src_dir, dst_dir, ratio):
    # 创建目标目录（如果不存在）
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 获取源目录下所有的子目录和文件名（不包括子目录下的内容）
    subdirs, files = [], []
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
        elif os.path.isfile(item_path) and item.endswith(".wav"):
            files.append(item)

    # 如果源目录下没有任何wav文件，则报错并退出函数
    if len(files) > 0:
        # 计算需要抽取的wav文件数量
        num_files = int(len(files) * ratio)
        num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))

        # 随机打乱文件名列表，并取出前num_files个作为抽取结果
        np.random.shuffle(files)
        selected_files = files[:num_files]

        # 创建一个进度条对象，用于显示程序的运行进度
        pbar = tqdm.tqdm(total=num_files)

        # 遍历抽取结果中的每个文件名
        for file in selected_files:
            # 拼接源文件和目标文件的完整路径
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            # 检查源文件的时长是否大于2秒
            if check_duration(src_file):
                # 如果是，则剪切源文件到目标目录中
                shutil.move(src_file, dst_file)
                # 更新进度条
                pbar.update(1)
            else:
                # 如果不是，则打印源文件的文件名，并跳过该文件
                print(f"Skipped {src_file} because its duration is less than 2 seconds.")
        
        # 关闭进度条
        pbar.close()
    else:
        print(f"Error: No wav files found in {src_dir}")

    # 遍历源目录下所有的子目录（如果有）
    for subdir in subdirs:
        # 拼接子目录在源目录和目标目录中的完整路径
        src_subdir = os.path.join(src_dir, subdir)
        dst_subdir = os.path.join(dst_dir, subdir)
        # 递归地调用本函数，对子目录中的wav文件进行同样的操作，保留数据结构
        split_data(src_subdir, dst_subdir, ratio)

# 定义主函数，用于获取用户输入并调用上述函数

def main():
    root_dir = os.path.abspath('.')
    dst_dir = root_dir + "/data/val/audio"
    # 抽取比例，默认为1
    ratio = float(SAMPLE_RATE) / 100

    # 固定源目录为根目录下/data/train/audio目录
    src_dir = root_dir + "/data/train/audio"

    # 调用split_data函数，对源目录中的wav文件进行抽取，并剪切到目标目录中，保留数据结构
    split_data(src_dir, dst_dir, ratio)

# 如果本模块是主模块，则执行主函数
if __name__ == "__main__":
    main()
