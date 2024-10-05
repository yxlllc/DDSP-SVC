import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import re

import numpy as np
from tqdm import tqdm
import yaml

from logger import utils


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=False, help="path to the config file"
    )
    return parser.parse_args(args=args, namespace=namespace)


def split_array(arr, n):
    # 使用列表推导式按每块n个元素进行分块
    chunks = [arr[i : i + n] for i in range(0, len(arr), n)]
    last_chunk_length = len(chunks[-1]) if chunks else 0  # 获取最后一项的长度
    return chunks, last_chunk_length


def make_chunk(
    chunk: list[str],
    path: str,
    idx: int,
) -> list[dict]:
    data = []
    for features_path in chunk:
        ft = np.load(features_path)
        ft_data = {}
        for key in ft.files:
            if key not in [
                "mel",
                "aug_mel",
                "units",
            ]:
                ft_data[key] = ft[key]

        ft_data["features_path"] = features_path
        dirname_split = os.path.dirname(features_path).split(os.sep)[-1]
        spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
        if spk_id < 1:
            raise ValueError(
                " [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk "
            )
        ft_data["spk_id"] = spk_id
        data.append(ft_data)
    np.savez_compressed(os.path.join(path, f"chunk_{idx}.npz"), data=data)


if __name__ == "__main__":
    cmd = parse_args()
    config = cmd.config
    if not config:
        config = "./configs/reflow.yaml"
    print(config)

    # 读取配置文件
    args = utils.load_config(config)

    filelist = utils.traverse_dir(
        os.path.join(args.data.train_path, "features"),
        extensions=["npz"],
        is_ext=True,
    )

    chunks, last_chunk_length = split_array(filelist, args.data.big_chunk_size)
    path_chunksdir = os.path.join(args.data.train_path, "chunks")
    os.makedirs(path_chunksdir, exist_ok=True)

    yaml.dump(
        {
            "num_features": len(filelist),
            "num_chunks": len(chunks),
            "num_features_per_chunk": args.data.big_chunk_size,
            "last_chunk_length": last_chunk_length,
        },
        open(os.path.join(path_chunksdir, "info.yaml"), "w"),
    )
    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, chunk in enumerate(chunks):
            futures.append(executor.submit(make_chunk, chunk, path_chunksdir, idx))

        for future in tqdm(futures):
            future.result()
