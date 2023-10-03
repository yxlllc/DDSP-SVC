Language: [English](./README.md) **简体中文**

# DDSP-SVC

## (4.0 升级) 新的 DDSP 级联扩散模型

安装依赖，数据准备，配置编码器（hubert 或者 contentvec) ，声码器 (nsf-hifigan) 与音高提取器 (RMVPE) 的环节与训练纯 DDSP 模型相同 （见下面的章节）。

我们提供了一个预训练模型：
<https://huggingface.co/datasets/ms903/DDSP-SVC-4.0/resolve/main/pre-trained-model/model_0.pt> (使用 'contentvec768l12' 编码器)

将名为`model_0.pt`的预训练模型, 放到`diffusion-new.yaml`里面 "expdir: exp/\*\*\*\*\*" 参数指定的模型导出文件夹内, 没有就新建一个, 程序会自动加载该文件夹下的预训练模型。

（1）预处理：

```bash
python preprocess.py -c configs/diffusion-new.yaml
```

（2）训练级联模型 (只训练一个模型)：

```bash
python train_diff.py -c configs/diffusion-new.yaml
```

注：fp16 训练暂时有问题，fp32 和 bf16 是可以正常训练的。

（3）非实时推理：

```bash
python main_diff.py -i <input.wav> -diff <diff_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -speedup <speedup> -method <method> -kstep <kstep>
```

4.0 版本模型内置了 DDSP 模型，因此不需要使用 -ddsp 指定外部 DDSP 模型， 其他选项与 3.0 版本模型含义相同，但 kstep 需要小于等于配置文件中的 `k_step_max`，建议保持相等 （默认是 100）。

（4）实时 GUI :

```bash
python gui_diff.py
```

注：你需要在 GUI 的右手边加载 4.0 版本模型。

## （3.0 升级）浅扩散模型 （DDSP + Diff-SVC 重构版）

![Diagram](diagram.png)

安装依赖，数据准备，配置编码器（hubert 或者 contentvec) ，声码器 (nsf-hifigan) 与音高提取器 (RMVPE) 的环节与训练纯 DDSP 模型相同 （见下面的章节）。

因为扩散模型更难训练，我们提供了一些预训练模型：

<https://huggingface.co/datasets/ms903/Diff-SVC-refactor-pre-trained-model/blob/main/hubertsoft_fix_pitch_add_vctk_500k/model_0.pt> (使用 'hubertsoft' 编码器)

<https://huggingface.co/datasets/ms903/Diff-SVC-refactor-pre-trained-model/blob/main/fix_pitch_add_vctk_600k/model_0.pt> (使用 'contentvec768l12' 编码器)

将名为`model_0.pt`的预训练模型, 放到`diffusion.yaml`里面 "expdir: exp/\*\*\*\*\*" 参数指定的模型导出文件夹内, 没有就新建一个, 程序会自动加载该文件夹下的预训练模型。

（1）预处理：

```bash
python preprocess.py -c configs/diffusion.yaml
```

这个预处理也能用来训练 DDSP 模型，不用预处理两遍（但需要保证 yaml 里面的 data 下面的参数均一致）

（2）训练扩散模型：

```bash
python train_diff.py -c configs/diffusion.yaml
```

（3）训练 DDSP 模型：

```bash
python train.py -c configs/combsub.yaml
```

如上所述，可以不需要重新预处理，但请检查 combsub.yaml 与 diffusion.yaml 是否参数匹配。说话人数 n_spk 可以不一致，但是尽量用相同的编号表示相同的说话人（推理更简单）。

（4）非实时推理：

```bash
python main_diff.py -i <input.wav> -ddsp <ddsp_ckpt.pt> -diff <diff_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -diffid <diffusion_speaker_id> -speedup <speedup> -method <method> -kstep <kstep>
```

speedup 为加速倍速，method 为 ddim, pndm, dpm-solver 或 unipc, kstep 为浅扩散步数，diffid 为扩散模型的说话人 id，其他参数与 main.py 含义相同。

合理的 kstep 约为 100\~300，speedup 超过 20 时可能将感知到音质损失。

如果训练时已经用相同的编号表示相同的说话人，则 -diffid 可以为空，否则需要指定 -diffid 选项。

如果 -ddsp 为空，则使用纯扩散模型 ，此时以输入源的 mel 进行浅扩散，若进一步 -kstep 为空，则进行完整深度的高斯扩散。

程序会自动检查 DDSP 模型和扩散模型的参数是否匹配 （采样率，帧长和编码器），不匹配会忽略加载 DDSP 模型并进入高斯扩散模式。

（5）实时 GUI :

```bash
python gui_diff.py
```

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

## 1. 安装依赖

1.  安装 PyTorch：我们推荐从 [\*\*PyTorch 官方网站 \*\*](https://pytorch.org/) 下载 PyTorch.

2.  安装依赖

```bash
pip install -r requirements.txt
```

注： 仅在 python 3.8 (windows) + torch 1.9.1 + torchaudio 0.6.0 测试过代码，太旧或太新的依赖可能会报错。

更新：python 3.8 (windows) + cuda 11.8 + torch 2.0.0 + torchaudio 2.0.1 可以运行，训练速度更快了。

## 2. 配置预训练模型

- 特征编码器 (可只选其一)：

(1) 下载预训练 [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 编码器并将其放到 `pretrain/contentvec` 文件夹。

(2) 下载预训练 [HubertSoft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) 编码器并将其放到 `pretrain/hubert` 文件夹，同时修改配置文件。

- 声码器或增强器：

下载预训练 [NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip) 声码器并解压至 `pretrain/` 文件夹。

- 音高提取器:

下载预训练 [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) 提取器并解压至 `pretrain/` 文件夹

## 3. 预处理

### 1. 配置训练数据集和验证数据集

#### 1.1 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`。

#### 1.2 程序随机选择：

运行`python draw.py`,程序将帮助你挑选验证集数据（可以调整 `draw.py` 中的参数修改抽取文件的数量等参数）。

#### 1.3 文件夹结构目录展示：

- 单人物目录结构：

<!---->

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

- 多人物目录结构：

<!---->

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

### 2. 样例合成器模型训练

1.  训练基于梳齿波减法合成器的模型 (**推荐**)：

```bash
python preprocess.py -c configs/combsub.yaml
```

1.  训练基于正弦波加法合成器的模型：

```bash
python preprocess.py -c configs/sins.yaml
```

1.  您可以在预处理之前修改配置文件 `config/<model_name>.yaml`，默认配置适用于 GTX-1660 显卡训练 44.1khz 高采样率合成器。

2.  如果要训练扩散模型，见上述 3.0 或 4.0 章节

### 3. 备注：

1.  请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。（可选：使用 Adobe Audition™ 的响度匹配功能可以一次性完成重采样修改声道和响度匹配。）

2.  训练数据集的音频切片总数建议为约 1000 个，另外长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

3.  验证集的音频切片总数建议为 10 个左右，不要放太多，不然验证过程会很慢。

4.  如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'rmvpe'.

5.  配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**。

## 4. 训练

```bash
# 以训练 combsub 模型为例
python train.py -c configs/combsub.yaml
```

1.  训练其他模型方法类似。

2.  可以随时中止训练，然后运行相同的命令来继续训练。

3.  微调 (finetune)：在中止训练后，重新预处理新数据集或更改训练参数（batchsize、lr 等），然后运行相同的命令。

## 5. 可视化

```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```

第一次验证 (validation) 后，在 TensorBoard 中可以看到合成后的测试音频。

注：TensorBoard 中的测试音频是 DDSP-SVC 模型的原始输出，并未通过增强器增强。 如果想测试模型使用增强器的合成效果（可能具有更高的合成质量），请使用下一章中描述的方法。

## 6. 非实时变声

1.  （**推荐**）使用预训练声码器增强 DDSP 的输出结果：

```bash
# 默认 enhancer_adaptive_key = 0 正常音域范围内将有更高的音质
# 设置 enhancer_adaptive_key > 0 可将增强器适配于更高的音域
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -e true -eak <enhancer_adaptive_key (semitones)>
```

1.  DDSP 的原始输出结果：

```bash
# 速度快，但音质相对较低（像您在tensorboard里听到的那样）
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -e false -id <speaker_id>
```

1.  关于 f0 提取器、响应阈值及其他参数，参见:

```bash
python main.py -h
```

1.  如果要使用混合说话人（捏音色）功能，增添 “-mix” 选项来设计音色，下面是个例子：

```bash
# 将1号说话人和2号说话人的音色按照0.5:0.5的比例混合
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -e true -eak 0
```

## 7. 实时变声

用以下命令启动简易操作界面:

```bash
python gui.py
```

该前端使用了滑动窗口，交叉淡化，基于 SOLA 的拼接和上下文语义参考等技术，在低延迟和资源占用的情况下可以达到接近非实时合成的音质。

更新：现在加入了基于相位声码器的衔接算法，但是大多数情况下 SOLA 算法已经具有足够高的拼接音质，所以它默认是关闭状态。如果您追求极端的低延迟实时变声音质，可以考虑开启它并仔细调参，有概率音质更高。但大量测试发现，如果交叉淡化时长大于 0.1 秒，相位声码器反而会造成音质明显劣化。

## 8. 感谢

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [soft-vc](https://github.com/bshall/soft-vc)

- [ContentVec](https://github.com/auspicious3000/contentvec)

- [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)

- [Diff-SVC](https://github.com/prophesier/diff-svc)
