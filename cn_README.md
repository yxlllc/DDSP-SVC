Language: [English](./README.md) **简体中文**

# DDSP-SVC

## 0.简介

DDSP-SVC 是一个新的开源歌声转换项目，致力于开发可以在个人电脑上普及的自由 AI 变声器软件。

相比于著名的 [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc), 它训练和合成对电脑硬件的要求要低的多，并且训练时长有数量级的缩短，和 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的训练速度接近。

另外在进行实时变声时，本项目的硬件资源消耗显著低于 SO-VITS-SVC , 但可能略高于 RVC 最新版本。

虽然 DDSP 的原始合成质量不是很理想（训练时在 tensorboard 中可以听到原始输出），但在使用基于预训练声码器的增强器（老版本）或使用浅扩散模型（新版本）增强音质后，对于部分数据集可以达到不亚于 SOVITS-SVC 和 RVC 的合成质量。

老版本的模型仍然是兼容的，以下章节是老版本的使用说明。新版本部分操作是相同的，见之前章节。

免责声明：请确保仅使用**合法获得的授权数据**训练 DDSP-SVC 模型，不要将这些模型及其合成的任何音频用于非法目的。 本库作者不对因使用这些模型检查点和音频而造成的任何侵权，诈骗等违法行为负责。

1.1 更新：支持多说话人和音色混合。

2.0 更新：开始支持实时 vst 插件，并优化了 combsub 模型， 训练速度极大提升。旧的 combsub 模型仍然兼容，可用 combsub-old.yaml 训练，sins 模型不受影响，但由于训练速度远慢于 combsub, 目前版本已经不推荐使用。

3.0 更新：由于作者删库 vst 插件取消支持，转为使用独立的实时变声前端；支持多种编码器，并将 contentvec768l12 作为默认编码器；引入浅扩散模型，合成质量极大提升。

4.0 更新：支持最先进的 RMVPE 音高提取器，联合训练 DDSP 与扩散模型，提升推理与训练速度，进一步提升合成质量。

5.0 更新：支持更快速的 FCPE 音高提取器，改进 DDSP 模型与扩散模型，提升推理与训练速度，进一步提升合成质量。

6.1 更新：改进 DDSP 模型, 并采用整流流（Rectified-Flow）模型替换扩散模型，进一步提升合成质量，旧模型不再兼容

## 1. 安装依赖

1. 安装 PyTorch：我们推荐从 [**PyTorch 官方网站 **](https://pytorch.org/) 下载 PyTorch.

2. 安装依赖

```bash
pip install -r requirements.txt
```

python 3.8 (windows) + cuda 11.8 + torch 2.0.0 + torchaudio 2.0.1 可以运行

## 2. 配置预训练模型

* 特征编码器 (可只选其一)：

(1) 下载预训练 [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 编码器并将其放到 `pretrain/contentvec` 文件夹。

(2) 下载预训练 [HubertSoft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) 编码器并将其放到 `pretrain/hubert` 文件夹，同时修改配置文件。

* 声码器：

下载并解压预训练 [NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip) 声码器

或者使用 https://github.com/openvpi/SingingVocoders 微调声码器以获得更高音质。

然后重命名权重文件并放置在配置文件中 'vocoder.ckpt' 参数指定的位置，默认值是 `pretrain/nsf_hifigan/model`。

声码器的 'config.json' 需要在同目录，比如 `pretrain/nsf_hifigan/config.json`。

* 音高提取器:

下载预训练 [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) 提取器并解压至 `pretrain/` 文件夹

## 3. 预处理

### 3.1. 配置训练数据集和验证数据集

1. 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`。

2. 程序随机选择：

运行`python draw.py`,程序将帮助你挑选验证集数据（可以调整 `draw.py` 中的参数修改抽取文件的数量等参数）。

3. 文件夹结构目录展示：

* 单人物目录结构：

```
data
├─ train
│    ├─ audio
│    │    ├─ aaa.wav
│    │    ├─ bbb.wav
│    │    └─ ....wav
├─ val
│    ├─ audio
│    │    ├─ eee.wav
│    │    ├─ fff.wav
│    │    └─ ....wav
```

* 多人物目录结构：

```
data
├─ train
│    ├─ audio
│    │    ├─ 1
│    │    │   ├─ aaa.wav
│    │    │   ├─ bbb.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ccc.wav
│    │    │   ├─ ddd.wav
│    │    │   └─ ....wav
│    │    └─ ...
|
├─ val
|    ├─ audio
│    │    ├─ 1
│    │    │   ├─ eee.wav
│    │    │   ├─ fff.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ggg.wav
│    │    │   ├─ hhh.wav
│    │    │   └─ ....wav
│    │    └─ ...
```

### 3.2. 执行预处理

```bash
python preprocess.py -c configs/reflow.yaml
```

1. 默认配置适用于 GTX-1660 显卡训练 44.1khz 高采样率合成器。

2. 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。（可选：使用 Adobe Audition™ 的响度匹配功能可以一次性完成重采样修改声道和响度匹配。）

3. 训练数据集的音频切片总数建议为约 1000 个，另外长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

4. 验证集的音频切片总数建议为 10 个左右，不要放太多，不然验证过程会很慢。

5. 如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'rmvpe'.

6. 配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**。

## 4. 训练

```bash
python train_reflow.py -c configs/reflow.yaml
```

1. 训练开始后，每 ‘interval_val’ 步临时保存一个权重，每 ‘interval_force_save’ 步永久保存一个权重，可根据情况修改这两个配置项。

2. 可以随时中止训练，然后运行相同的命令来从最新保存的权重开始继续训练。

3. 微调 (finetune)：在中止训练后，重新预处理新数据集或更改训练参数（batchsize、lr 等），然后运行相同的命令。

## 5. 可视化

```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```

第一次验证 (validation) 后，在 TensorBoard 中可以看到合成的测试音频。

## 6. 非实时变声

```bash
python main_reflow.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -step <infer_step> -method <method> -ts <t_start>
```

'infer_step' 为 rectified-flow ODE 的采样步数，'method' 为 'euler' 或 'rk4'，'t_start' 为 ODE 的起始时间点，需要大于或等于配置文件中的 `t_start`，建议保持相等（默认为 0.7）。

如果要使用混合说话人（捏音色）功能，增添 “-mix” 选项来设计音色，下面是个例子：

```bash
# 将1号说话人和2号说话人的音色按照 0.5:0.5 的比例混合
python main_reflow.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}"
```

关于 f0 提取器、响应阈值及其他参数，参见:

```bash
python main_reflow.py -h
```

## 7. 实时变声

用以下命令启动简易操作界面:

```bash
python gui_reflow.py
```

该前端使用了滑动窗口，交叉淡化，基于 SOLA 的拼接和上下文语义参考等技术，在低延迟和资源占用的情况下可以达到接近非实时合成的音质。

## 8. 感谢

* [ddsp](https://github.com/magenta/ddsp)

* [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

* [soft-vc](https://github.com/bshall/soft-vc)

* [ContentVec](https://github.com/auspicious3000/contentvec)

* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)

* [Diff-SVC](https://github.com/prophesier/diff-svc)

* [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)

