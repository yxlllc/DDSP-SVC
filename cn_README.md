Language: [English](./README.md) **简体中文**
# DDSP-SVC
<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/ddsp_logo.png" width="200px" alt="logo"></img>
</div>
基于 DDSP（可微分数字信号处理）的实时端到端歌声转换系统

## （3.0 - 实验性）浅扩散模型 （DDSP + Diff-SVC 重构版）
数据准备，配置编码器与声码器的环节与训练纯 DDSP 模型相同。
预处理：
```bash
python preprocess.py -c configs/diffusion.yaml
```
这个预处理也能用来训练 DDSP 模型，不用预处理两遍（但需要保证 yaml 里面的 data 下面的参数均一致）

训练扩散模型：
```bash
python train_diff.py -c configs/diffusion.yaml
```
训练 DDSP 模型：
```bash
python train.py -c configs/combsub.yaml
```
如上所述，可以不需要重新预处理，但请检查 combsub.yaml 与 diffusion.yaml 是否参数匹配。说话人数 n_spk 可以不一致，但是尽量用相同的编号表示相同的说话人（推理更简单）。

非实时推理：
```bash
python main_diff.py -i <input.wav> -ddsp <ddsp_ckpt.pt> -diff <diff_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -diffid <diffusion_speaker_id> -speedup <speedup> -method <method> -kstep <kstep>
```
speedup 为加速倍速，method 为 pndm 或者 dpm-solver, kstep为浅扩散步数，diffid 为扩散模型的说话人id，其他参数与 main.py 含义相同。

如果训练时已经用相同的编号表示相同的说话人，则 -diffid 可以为空，否则需要指定 -diffid 选项。

如果 -ddsp 为空，则使用纯扩散模型 ，此时以输入源的 mel 进行浅扩散，若进一步 -kstep 为空，则进行完整深度的高斯扩散。

程序会自动检查 DDSP 模型和扩散模型的参数是否匹配 （采样率，帧长和编码器），不匹配会忽略加载 DDSP 模型并进入高斯扩散模式。

实时 GUI :
```bash
python gui_diff.py
```


## 0.简介
DDSP-SVC 是一个新的开源歌声转换项目，致力于开发可以在个人电脑上普及的自由 AI 变声器软件。

相比于比较著名的 [Diff-SVC](https://github.com/prophesier/diff-svc) 和 [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc), 它训练和合成对电脑硬件的要求要低的多，并且训练时长有数量级的缩短。另外在进行实时变声时，本项目的硬件资源显著低于 SO-VITS-SVC，而 Diff-SVC 合成太慢几乎无法进行实时变声。

虽然 DDSP 的原始合成质量不是很理想（训练时在 tensorboard 中可以听到原始输出），但在使用基于预训练声码器的增强器增强音质后，对于部分数据集可以达到接近 SOVITS-SVC 的合成质量。

如果训练数据的质量非常高，可能仍然 Diff-SVC 将拥有最高的合成质量。在`samples`文件夹中包含合成示例，相关模型检查点可以从仓库发布页面下载。

免责声明：请确保仅使用**合法获得的授权数据**训练 DDSP-SVC 模型，不要将这些模型及其合成的任何音频用于非法目的。 本库作者不对因使用这些模型检查点和音频而造成的任何侵权，诈骗等违法行为负责。

1.1 更新：支持多说话人和音色混合。

2.0 更新：开始支持实时 vst 插件，并优化了 combsub 模型， 训练速度极大提升。旧的 combsub 模型仍然兼容，可用 combsub-old.yaml 训练，sins 模型不受影响，但由于训练速度远慢于 combsub, 目前版本已经不推荐使用。

## 1. 安装依赖
1. 安装PyTorch：我们推荐从 [**PyTorch 官方网站 **](https://pytorch.org/) 下载 PyTorch.

2. 安装依赖
```bash
pip install -r requirements.txt 
```
注： 仅在 python 3.8 (windows) + torch 1.9.1 + torchaudio 0.6.0 测试过代码，太旧或太新的依赖可能会报错。

更新：python 3.8 (windows) + cuda 11.8 + torch 2.0.0 + torchaudio 2.0.1 可以运行，训练速度更快了。
## 2. 配置预训练模型
- **(必要操作)** 下载预训练 [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) 编码器并将其放到 `pretrain/hubert` 文件夹。
  - 更新：现在支持 ContentVec 编码器了。你可以下载预训练 [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 编码器替代 HubertSoft 编码器并修改配置文件以使用它。
- 从 [DiffSinger 社区声码器项目](https://openvpi.github.io/vocoders) 下载基于预训练声码器的增强器，并解压至 `pretrain/` 文件夹。
  -  注意：你应当下载名称中带有`nsf_hifigan`的压缩文件，而非`nsf_hifigan_finetune`。
## 3. 预处理

### 1. 配置训练数据集和验证数据集

#### 1.1 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`。

#### 1.2 程序随机选择：

运行`python draw.py`,程序将帮助你挑选验证集数据（可以调整 `draw.py` 中的参数修改抽取文件的数量等参数）。

#### 1.3文件夹结构目录展示：
- 单人物目录结构：

```
data
├─ train
│    ├─ audio
│    │    ├─ aaa.wav
│    │    ├─ bbb.wav
│    │    └─ ....wav
│    └─ val
│    │    ├─ eee.wav
│    │    ├─ fff.wav
│    │    └─ ....wav
 ```
- 多人物目录结构：

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
│    └─ val
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
### 2. 样例合成器模型训练
1. 训练基于梳齿波减法合成器的模型 (**推荐**)：

```bash
python preprocess.py -c configs/combsub.yaml
```

2. 训练基于正弦波加法合成器的模型：

```bash
python preprocess.py -c configs/sins.yaml
```

3. 您可以在预处理之前修改配置文件 `config/<model_name>.yaml`，默认配置适用于GTX-1660 显卡训练 44.1khz 高采样率合成器。

### 3. 备注：
1. 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。（可选：使用Adobe Audition™的响度匹配功能可以一次性完成重采样修改声道和响度匹配。）

2. 训练数据集的音频切片总数建议为约 1000 个，另外长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

3. 验证集的音频切片总数建议为 10 个左右，不要放太多，不然验证过程会很慢。

4. 如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'crepe'。crepe 算法的抗噪性最好，但代价是会极大增加数据预处理所需的时间。

5. 配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**。
## 4. 训练

### 1. 不使用预训练数据进行训练：
```bash
# 以训练 combsub 模型为例 
python train.py -c configs/combsub.yaml
```
1. 训练其他模型方法类似。

2. 可以随时中止训练，然后运行相同的命令来继续训练。

3. 微调 (finetune)：在中止训练后，重新预处理新数据集或更改训练参数（batchsize、lr等），然后运行相同的命令。
### 2. 使用预训练数据（底模）进行训练：
1. **使用预训练模型请修改配置文件中的 'n_spk' 参数为 '2' ,同时配置`train`目录结构为多人物目录，不论你是否训练多说话人模型。**
2. **如果你要训练一个更多说话人的模型，就不要下载预训练模型了。**
3. 欢迎PR训练的多人底模 (请使用授权同意开源的数据集进行训练)。
4. 从[**这里**](https://github.com/yxlllc/DDSP-SVC/releases/download/2.0/opencpop+kiritan.zip)下载预训练模型，并将`model_300000.pt`解压到`.\exp\combsub-test\`中
5. 同不使用预训练数据进行训练一样，启动训练。
## 5. 可视化
```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```
第一次验证 (validation) 后，在 TensorBoard 中可以看到合成后的测试音频。

注：TensorBoard 中的测试音频是 DDSP-SVC 模型的原始输出，并未通过增强器增强。 如果想测试模型使用增强器的合成效果（可能具有更高的合成质量），请使用下一章中描述的方法。
## 6. 非实时变声
1. （**推荐**）使用预训练声码器增强 DDSP 的输出结果：
```bash
# 默认 enhancer_adaptive_key = 0 正常音域范围内将有更高的音质
# 设置 enhancer_adaptive_key > 0 可将增强器适配于更高的音域
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -e true -eak <enhancer_adaptive_key (semitones)>
```
2. DDSP 的原始输出结果：
```bash
# 速度快，但音质相对较低（像您在tensorboard里听到的那样）
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -e false -id <speaker_id>
```
3. 关于 f0 提取器、响应阈值及其他参数，参见:

```bash
python main.py -h
```
4. 如果要使用混合说话人（捏音色）功能，增添 “-mix” 选项来设计音色，下面是个例子：
```bash
# 将1号说话人和2号说话人的音色按照0.5:0.5的比例混合
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -e true -eak 0
```
## 7. 实时变声
用以下命令启动简易操作界面:
```bash
python gui.py
```
该前端使用了滑动窗口，交叉淡化，基于SOLA 的拼接和上下文语义参考等技术，在低延迟和资源占用的情况下可以达到接近非实时合成的音质。

更新：现在加入了基于相位声码器的衔接算法，但是大多数情况下 SOLA 算法已经具有足够高的拼接音质，所以它默认是关闭状态。如果您追求极端的低延迟实时变声音质，可以考虑开启它并仔细调参，有概率音质更高。但大量测试发现，如果交叉淡化时长大于0.1秒，相位声码器反而会造成音质明显劣化。
## 8. 感谢
* [ddsp](https://github.com/magenta/ddsp)
* [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
* [soft-vc](https://github.com/bshall/soft-vc)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
