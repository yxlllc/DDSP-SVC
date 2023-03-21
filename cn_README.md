Language: [English](./README.md) ** 简体中文 **
# DDSP-SVC
<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/ddsp_logo.png" width="200px" alt="logo"></img>
</div>
基于 DDSP（可微分数字信号处理）的端到端歌声转换系统

## 0.简介
DDSP-SVC 是一个新的开源歌声转换项目，致力于开发可以在个人电脑上普及的自由 AI 变声器软件。

相比于比较著名的 [Diff-SVC](https://github.com/prophesier/diff-svc) 和 [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc), 它训练和合成对电脑硬件的要求要低的多，并且训练时长有数量级的缩短。

虽然 DDSP 的原始合成质量不是很理想（训练时在 tensorboard 中可以听到原始输出），但在使用基于预训练声码器的增强器增强音质后，对于部分数据集可以达到接近 SOVITS-SVC 的合成质量。

如果训练数据的质量非常高，可能仍然 Diff-SVC 将拥有最高的合成质量。在`samples`文件夹中包含合成示例，相关模型检查点可以从仓库发布页面下载。

免责声明：请确保仅使用**合法获得的授权数据**训练 DDSP-SVC 模型，不要将这些模型及其合成的任何音频用于非法目的。 本库作者不对因使用这些模型检查点和音频而造成的任何侵权，诈骗等违法行为负责。

1.1 更新：支持多说话人和音色混合
2.0 更新：开始支持实时 vst 插件，并优化了 combsub 模型， 训练速度极大提升。旧的 combsub 模型仍然兼容，可用 combsub-old.yaml 训练，sins 模型不受影响，但由于训练速度远慢于 combsub, 目前版本已经不推荐使用

## 1. 安装依赖
我们推荐从 [**PyTorch 官方网站 **](https://pytorch.org/) 下载 PyTorch.

接着运行
```bash
pip install -r requirements.txt 
```
注： 我只在 python 3.8 (windows) + pytorch 1.9.1 + torchaudio 0.6.0 测试过代码，太旧或太新的依赖可能会报错。
## 2. 配置预训练模型
- **(必要操作)** 下载预训练 [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) 编码器并将其放到 `pretrain/hubert` 文件夹.
-  从 [DiffSinger 社区声码器项目](https://openvpi.github.io/vocoders) 下载基于预训练声码器的增强器，并解压至 `pretrain/` 文件夹。
## 3. 预处理

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`.

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`.

你也可以运行
```bash
python draw.py
```
帮助你挑选验证集数据（可以调整 `draw.py` 中的参数修改抽取文件的数量等参数）

接着运行
```bash
python preprocess.py -c configs/combsub.yaml
```

训练基于梳齿波减法合成器的模型 (**推荐**)，或者运行

```bash
python preprocess.py -c configs/sins.yaml
```
训练基于正弦波加法合成器的模型。

您可以在预处理之前修改配置文件 `config/<model_name>.yaml`。

默认配置适用于GTX-1660 显卡训练 44.1khz 高采样率合成器。

注 1: 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。

注 2：训练数据集的音频切片总数建议为约 1000 个，另外长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

注 3：验证集的音频切片总数建议为 10 个左右，不要放太多，不然验证过程会很慢。

注4：如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'crepe'。crepe 算法的抗噪性最好，但代价是会极大增加数据预处理所需的时间。

更新：现在支持多说话人训练了，配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**，目录结构如下所示：
```bash
# 训练集
# 第1个说话人
data/train/audio/1/aaa.wav
data/train/audio/1/bbb.wav
...
# 第2个说话人
data/train/audio/2/ccc.wav
data/train/audio/2/ddd.wav
...

# 验证集
# 第1个说话人
data/val/audio/1/eee.wav
data/val/audio/1/fff.wav
...
# 第2个说话人
data/val/audio/2/ggg.wav
data/val/audio/2/hhh.wav
...
```
当 'n_spk' =1 时，之前**单说话人**模型的目录结构仍然支持，即：

```bash
# 训练集
data/train/audio/aaa.wav
data/train/audio/bbb.wav
...
# 验证集
data/val/audio/ccc.wav
data/val/audio/ddd.wav
...
```
## 4. 训练
```bash
# 以训练 combsub 模型为例 
python train.py -c configs/combsub.yaml
```
训练其他模型方法类似。

您可以随时中止训练，然后运行相同的命令来继续训练。

您也可以在中止训练后，重新预处理新数据集或更改训练参数（batchsize、lr等），然后运行相同的命令，就可以对模型进行微调 (finetune)。
## 5. 可视化
```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```
第一次验证 (validation) 后，在 TensorBoard 中可以看到合成后的测试音频。

注：TensorBoard 中的测试音频是 DDSP-SVC 模型的原始输出，并未通过增强器增强。 如果想测试模型使用增强器的合成效果（可能具有更高的合成质量），请使用下一章中描述的方法。
## 6. 测试
（**推荐**）使用预训练声码器增强 DDSP 的输出结果：
```bash
# 默认 enhancer_adaptive_key = 0 正常音域范围内将有更高的音质
# 设置 enhancer_adaptive_key > 0 可将增强器适配于更高的音域
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -e true -eak <enhancer_adaptive_key (semitones)>
```
 DDSP 的原始输出结果：
```bash
# 速度快，但音质相对较低（像您在tensorboard里听到的那样）
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -e false -id <speaker_id>
```
关于 f0 提取器和响应阈值的其他选项，参见:

```bash
python main.py -h
```
更新： 现在支持混合说话人（捏音色）了。您可以使用 “-mix” 选项来设计属于您自己的音色，下面是个例子：
```bash
# 将1号说话人和2号说话人的音色按照0.5:0.5的比例混合
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -e true -eak 0
```
## 7. HTTP 服务器 和 VST 支持
用以下命令启动服务器
```bash
# 配置在这个 python 文件里面，见注释
python flask_api.py
```
当前支持的 VST 前端:
https://github.com/zhaohui8969/VST_NetProcess-

## 8. 感谢
* [ddsp](https://github.com/magenta/ddsp)
* [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
* [soft-vc](https://github.com/bshall/soft-vc)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
