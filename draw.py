import argparse
import numpy as np
import tqdm
import os
import shutil

import soundfile as sf

WAV_MIN_LENGTH = 2    # wav文件的最短时长 / The minimum duration of wav files
SAMPLE_MIN = 2    # 抽取的文件数量下限 / The lower limit of the number of files to be extracted
SAMPLE_MAX = 10    # 抽取的文件数量上限 / The upper limit of the number of files to be extracted


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    root_dir = os.path.abspath('.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        default=root_dir + "/data/train/audio", # 固定源目录为根目录下/data/train/audio目录
        help="directory where contains train dataset"
    )
    parser.add_argument(
        "-v",
        "--val",
        type=str,
        default=root_dir + "/data/val/audio",
        help="directory where contains validate dataset"
    )
    parser.add_argument(
        "-r",
        "--sample_rate",
        type=float,
        default=1,
        help="The percentage of files to be extracted"  # 抽取文件数量的百分比
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        required=False,
        nargs="*",
        default=["wav", "flac"],
        help="list of using file extensions, e.g.) -f wav flac ..."
    )
    return parser.parse_args(args=args, namespace=namespace)


# 定义一个函数，用于检查wav文件的时长是否大于最短时长
def check_duration(wav_file):
    # 打开wav文件
    f = sf.SoundFile(wav_file)
    # 获取帧数和帧率
    frames = f.frames
    rate = f.samplerate
    # 计算时长（秒）
    duration = frames / float(rate)
    # 关闭文件
    f.close()
    # 返回时长是否大于最短时长的布尔值
    return duration > WAV_MIN_LENGTH

# 定义一个函数，用于从给定的目录中随机抽取一定比例的wav文件，并剪切到另一个目录中，保留数据结构
def split_data(src_dir, dst_dir, ratio, extensions):
    # 创建目标目录（如果不存在）
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 获取源目录下所有的子目录和文件名
    subdirs, files, subfiles = [], [], []
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isfile(subitem_path) and any([subitem.endswith(f".{ext}") for ext in extensions]):
                    subfiles.append(subitem)
        elif os.path.isfile(item_path) and any([item.endswith(f".{ext}") for ext in extensions]):
            files.append(item)

    # 如果源目录下没有任何wav文件，则报错并退出函数
    if len(files) == 0:
        if len(subfiles) == 0:
            print(f"Error: No wav files found in {src_dir}")
            return
    
    # 计算需要抽取的wav文件数量
    num_files = int(len(files) * ratio)
    num_files = max(SAMPLE_MIN, min(SAMPLE_MAX, num_files))

    # 随机打乱文件名列表，并取出前num_files个作为抽取结果
    np.random.shuffle(files)
    selected_files = files[:num_files]
    
    # 创建一个进度条对象，用于显示程序的运行进度
    pbar = tqdm.tqdm(total=num_files)

    # 遍历抽取结果中的每个文件名，检查是否大于2秒
    for file in selected_files:
        src_file = os.path.join(src_dir, file)
        # 检查源文件的时长是否大于2秒，如果不是，则打印源文件的文件名，并跳过该文件
        if not check_duration(src_file):
            print(f"Skipped {src_file} because its duration is less than 2 seconds.")
            continue
        # 拼接源文件和目标文件的完整路径，移动文件，并更新进度条
        dst_file = os.path.join(dst_dir, file)
        shutil.move(src_file, dst_file)
        pbar.update(1)

    pbar.close()

    # 遍历源目录下所有的子目录（如果有）
    for subdir in subdirs:
        # 拼接子目录在源目录和目标目录中的完整路径
        src_subdir = os.path.join(src_dir, subdir)
        dst_subdir = os.path.join(dst_dir, subdir)
        # 递归地调用本函数，对子目录中的wav文件进行同样的操作，保留数据结构
        split_data(src_subdir, dst_subdir, ratio, extensions)

# 定义主函数，用于获取用户输入并调用上述函数

def main(cmd):
    dst_dir = cmd.val
    # 抽取比例，默认为1
    ratio = cmd.sample_rate / 100

    src_dir = cmd.train
    
    extensions = cmd.extensions

    # 调用split_data函数，对源目录中的wav文件进行抽取，并剪切到目标目录中，保留数据结构
    split_data(src_dir, dst_dir, ratio, extensions)

# 如果本模块是主模块，则执行主函数
if __name__ == "__main__":
    # parse commands
    cmd = parse_args()
    
    main(cmd)
