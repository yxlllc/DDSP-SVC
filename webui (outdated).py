import gradio as gr
import os,subprocess,yaml

class WebUI:
    def __init__(self) -> None:
        self.info=Info()
        self.opt_cfg_pth='configs/opt.yaml'
        self.main_ui()
    
    def main_ui(self):
        with gr.Blocks() as ui:
            gr.Markdown('## 一个便于训练和推理的DDSP-webui，每一步的说明在下面，可以自己展开看。')
            with gr.Tab("训练/Training"):
                gr.Markdown(self.info.general)
                with gr.Accordion('预训练模型说明',open=False):
                    gr.Markdown(self.info.pretrain_model)
                with gr.Accordion('数据集说明',open=False):
                    gr.Markdown(self.info.dataset)
                
                gr.Markdown('## 生成配置文件')
                with gr.Row():
                    self.batch_size=gr.Slider(minimum=2,maximum=60,value=24,label='Batch_size',interactive=True)
                    self.learning_rate=gr.Number(value=0.0005,label='学习率',info='和batch_size关系大概是0.0001:6')
                    self.f0_extractor=gr.Dropdown(['parselmouth', 'dio', 'harvest', 'crepe'],type='value',value='crepe',label='f0提取器种类',interactive=True)
                    self.sampling_rate=gr.Number(value=44100,label='采样率',info='数据集音频的采样率',interactive=True)
                    self.n_spk=gr.Number(value=1,label='说话人数量',interactive=True)
                with gr.Row():
                    self.device=gr.Dropdown(['cuda','cpu'],value='cuda',label='使用设备',interactive=True)
                    self.num_workers=gr.Number(value=2,label='读取数据进程数',info='如果你的设备性能很好，可以设置为0',interactive=True)
                    self.cache_all_data=gr.Checkbox(value=True,label='启用缓存',info='将数据全部加载以加速训练',interactive=True)
                    self.cache_device=gr.Dropdown(['cuda','cpu'],value='cuda',type='value',label='缓存设备',info='如果你的显存比较大，设置为cuda',interactive=True)
                self.bt_create_config=gr.Button(value='创建配置文件')
                
                gr.Markdown('## 预处理')
                with gr.Accordion('预训练说明',open=False):
                    gr.Markdown(self.info.preprocess)
                with gr.Row():
                    self.bt_open_data_folder=gr.Button('打开数据集文件夹')
                    self.bt_preprocess=gr.Button('开始预处理')
                gr.Markdown('## 训练')
                with gr.Accordion('训练说明',open=False):
                    gr.Markdown(self.info.train)
                with gr.Row():
                    self.bt_train=gr.Button('开始训练')
                    self.bt_visual=gr.Button('启动可视化')
                    gr.Markdown('启动可视化后[点击打开](http://127.0.0.1:6006)')
                
            with gr.Tab('推理/Inference'):
                with gr.Accordion('推理说明',open=False):
                    gr.Markdown(self.info.infer)
                with gr.Row():
                    self.input_wav=gr.Audio(type='filepath',label='选择待转换音频')
                    self.choose_model=gr.Textbox('exp/model_chino.pt',label='模型路径')
                with gr.Row():
                    self.keychange=gr.Slider(-24,24,value=0,step=1,label='变调')
                    self.id=gr.Number(value=1,label='说话人id')
                    self.enhancer_adaptive_key=gr.Number(value=0,label='增强器音区偏移',info='调高可以防止超高音（比如大于G5) 破音,但是低音效果可能会下降')
                with gr.Row():
                    self.bt_infer=gr.Button(value='开始转换')
                    self.output_wav=gr.Audio(type='filepath',label='输出音频')
                    
            self.bt_create_config.click(fn=self.create_config)
            self.bt_open_data_folder.click(fn=self.openfolder)
            self.bt_preprocess.click(fn=self.preprocess)
            self.bt_train.click(fn=self.training)
            self.bt_visual.click(fn=self.visualize)
            self.bt_infer.click(fn=self.inference,inputs=[self.input_wav,self.choose_model,self.keychange,self.id,self.enhancer_adaptive_key],outputs=self.output_wav)
        ui.launch(inbrowser=True,server_port=7858)
        
    def openfolder(self):
        try:
            os.startfile('data')
        except:
            print('Fail to open folder!')


    def create_config(self):
        with open('configs/combsub.yaml','r',encoding='utf-8') as f:
            cfg=yaml.load(f.read(),Loader=yaml.FullLoader)
        cfg['data']['f0_extractor']=str(self.f0_extractor.value)
        cfg['data']['sampling_rate']=int(self.sampling_rate.value)
        cfg['train']['batch_size']=int(self.batch_size.value)
        cfg['device']=str(self.device.value)
        cfg['train']['num_workers']=int(self.num_workers.value)
        cfg['train']['cache_all_data']=str(self.cache_all_data.value)
        cfg['train']['cache_device']=str(self.cache_device.value)
        cfg['train']['lr']=int(self.learning_rate.value)
        print('配置文件信息：'+str(cfg))
        with open(self.opt_cfg_pth,'w',encoding='utf-8') as f:
            yaml.dump(cfg,f)
        print('成功生成配置文件')

    
    def preprocess(self):
        preprocessing_process=subprocess.Popen('python -u preprocess.py -c '+self.opt_cfg_pth,stdout=subprocess.PIPE)
        while preprocessing_process.poll() is None:
            output=preprocessing_process.stdout.readline().decode('utf-8')
            print(output)
        print('预处理完成')
            
    def training(self):
        train_process=subprocess.Popen('python -u train.py -c '+self.opt_cfg_pth,stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output=train_process.stdout.readline().decode('utf-8')
            print(output)
        
            
    def visualize(self):
        tb_process=subprocess.Popen('tensorboard --logdir=exp --port=6006',stdout=subprocess.PIPE)
        while tb_process.poll() is None:
            output=tb_process.stdout.readline().decode('utf-8')
            print(output)
            
    def inference(self,input_wav:str,model:str,keychange,id,enhancer_adaptive_key):
        print(input_wav,model)
        output_wav='samples/'+ input_wav.replace('\\','/').split('/')[-1]
        cmd='python -u main.py -i '+input_wav+' -m '+model+' -o '+output_wav+' -k '+str(int(keychange))+' -id '+str(int(id))+' -e true -eak '+str(int(enhancer_adaptive_key))
        infer_process=subprocess.Popen(cmd,stdout=subprocess.PIPE)
        while infer_process.poll() is None:
            output=infer_process.stdout.readline().decode('utf-8')
            print(output)
        print('推理完成')
        return output_wav


class Info:
    def __init__(self) -> None:
        self.general='''
### 不看也没事，大致就是  
1.设置好配置之后点击创建配置文件  
2.点击‘打开数据集文件夹’，把数据集选个十个塞到data\\train\\val目录下面，剩下的音频全塞到data\\train\\audio下面  
3.点击‘开始预处理’等待执行完毕  
4.点击‘开始训练’和‘启动可视化’然后点击右侧链接  
'''
        self.pretrain_model="""
- **(必要操作)** 下载预训练 [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) 编码器并将其放到 `pretrain/hubert` 文件夹。
  - 更新：现在支持 ContentVec 编码器了。你可以下载预训练 [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 编码器替代 HubertSoft 编码器并修改配置文件以使用它。
- 从 [DiffSinger 社区声码器项目](https://openvpi.github.io/vocoders) 下载基于预训练声码器的增强器，并解压至 `pretrain/` 文件夹。
  -  注意：你应当下载名称中带有`nsf_hifigan`的压缩文件，而非`nsf_hifigan_finetune`。
        """
        self.dataset="""
### 1. 配置训练数据集和验证数据集

#### 1.1 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`。

#### 1.2 程序随机选择（**多人物时不可使用**）：

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
                            """
        self.preprocess='''
您可以在预处理之前修改配置文件 `config/<model_name>.yaml`，默认配置适用于GTX-1660 显卡训练 44.1khz 高采样率合成器。
### 备注：
1. 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。（可选：使用Adobe Audition™的响度匹配功能可以一次性完成重采样修改声道和响度匹配。）

2. 训练数据集的音频切片总数建议为约 1000 个，另外长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

3. 验证集的音频切片总数建议为 10 个左右，不要放太多，不然验证过程会很慢。

4. 如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'crepe'。crepe 算法的抗噪性最好，但代价是会极大增加数据预处理所需的时间。

5. 配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**。
        '''
        self.train='''
## 训练

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
        '''
        self.visualize='''
## 可视化
```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```
第一次验证 (validation) 后，在 TensorBoard 中可以看到合成后的测试音频。

注：TensorBoard 中的测试音频是 DDSP-SVC 模型的原始输出，并未通过增强器增强。
        '''
        self.infer='''
## 非实时变声
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
        '''




webui=WebUI()