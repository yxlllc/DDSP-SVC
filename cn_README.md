Language: [English](./README.md) ** 简体中文 **
# DDSP-SVC
<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/ddsp_logo.png" width="200px" alt="logo"></img>
</div>
基于 DDSP 的端到端语音转换

## 1. 安装依赖
我们推荐从 [**PyTorch 官方网站 **](https://pytorch.org/) 下载 PyTorch

接着运行
```bash
pip install -r requirements.txt 
```
## 2. 配置预训练模型
- **(必要操作)** 下载与训练 [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) 解码器并将其放到 `pretrain/hubert` 文件夹.
-  从 [DiffSinger 社区](https://openvpi.github.io/vocoders) 下载预训练声码器增强器，并放到 `pretrain/` 文件夹
## 3. 预处理

将所有的训练数据 (.wav 格式音频切片) 放到 `data/train/audio`.

将所有的验证数据 (.wav 格式音频切片) 放到 `data/val/audio`.

接着运行
```bash
python preprocess.py -c configs/combsub.yaml
```

训练 combtooth substractive 合成器


运行

```bash
python preprocess.py -c configs/sins.yaml
```
训练 sinusoids additive 合成器

您可以在预处理之前修改配置文件 `config/<model_name>.yaml`。

默认配置适用于 GTX-1660 显卡，训练 44.1khz 高采样率合成器。

注 1: 请保持所有音频剪辑的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。

注 2：训练数据集的音频剪辑总数建议约为 1000 个，特别是长音频，切成小段可以加快训练速度，但所有音频剪辑的持续时间不应少于 2 秒。如果音频分片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

注 3：验证数据集的音频剪辑总数建议为 10 个左右，不要放太多，不然验证会很慢。

## 4. 训练
```bash
# 训练 combsub 模型，作为例子 
python train.py -c configs/combsub.yaml
```

类似与训练其他模型

可以随时掐断训练，然后运行相同的命令来恢复训练。

可以掐断训练，然后重新预处理新数据集或更改训练参数（batchsize、lr等），然后运行相同的命令，就可以对模型进行微调。
## 5. 可视化
```bash
# 记得检查是否在训练
tensorboard --logdir=exp
```
## 6. 测试
```bash
# 会输出ddsp-svc的原始输出
# 速度快，但音质相对较低
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)>
```
```bash
# 快速但相对较低的音频质量使用预训练的基于声码器的增强器增强了输出
# 正常声音范围内的高音频质量，但速度较慢
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -e true
```
```bash
# 其他的自己看
python main.py -h
```

