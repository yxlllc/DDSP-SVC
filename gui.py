import PySimpleGUI as sg
import sounddevice as sd
import torch,librosa,threading,time
from enhancer import Enhancer
import numpy as np
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample


class SvcDDSP:
    def __init__(self, model_path, vocoder_based_enhancer, enhancer_adaptive_key, input_pitch_extractor,
                 f0_min, f0_max, threhold, spk_id, spk_mix_dict, enable_spk_id_cover):
        self.model_path = model_path
        self.vocoder_based_enhancer = vocoder_based_enhancer
        self.enhancer_adaptive_key = enhancer_adaptive_key
        self.input_pitch_extractor = input_pitch_extractor
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threhold = threhold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.spk_id = spk_id
        self.spk_mix_dict = spk_mix_dict
        self.enable_spk_id_cover = enable_spk_id_cover
        
        # load ddsp model
        self.model, self.args = load_model(self.model_path, device=self.device)
        
        # load units encoder
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            device=self.device)
        
        # load enhancer
        if self.vocoder_based_enhancer:
            self.enhancer = Enhancer(self.args.enhancer.type, self.args.enhancer.ckpt, device=self.device)

    def infer(self,  pitch_adjust, speaker_id, safe_prefix_pad_length,audio,sample_rate):
        print("Infering...")
        # load input
        #audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        # safe front silence
        if safe_prefix_pad_length > 0.03:
            silence_front = safe_prefix_pad_length - 0.03
        else:
            silence_front = 0
            
        # extract f0
        pitch_extractor = F0_Extractor(
            self.input_pitch_extractor,
            sample_rate,
            hop_size,
            float(self.f0_min),
            float(self.f0_max))
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)
        
        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(self.threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)
        
        # spk_id or spk_mix_dict
        if self.enable_spk_id_cover:
            spk_id = self.spk_id
        else:
            spk_id = speaker_id
        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)
        
        # forward and return the output
        with torch.no_grad():
            output, _, (s_h, s_n) = self.model(units, f0, volume, spk_id = spk_id, spk_mix_dict = self.spk_mix_dict)
            output *= mask
            if self.vocoder_based_enhancer:
                output, output_sample_rate = self.enhancer.enhance(
                                                                output, 
                                                                self.args.data.sampling_rate, 
                                                                f0, 
                                                                self.args.data.block_size,
                                                                adaptive_key = self.enhancer_adaptive_key,
                                                                silence_front = silence_front)
            else:
                output_sample_rate = self.args.data.sampling_rate

            output = output.squeeze().cpu().numpy()
            return output, output_sample_rate




class GUI:
    def __init__(self) -> None:
        self.flag_vc:bool=False#变声线程flag
        self.samplerate=44100#Hz
        self.block_time=1.5#s
        self.block_frame=0
        self.crossfade_frame=0
        self.fade_in_window:np.ndarray=None#crossfade计算用numpy数组
        self.fade_out_window:np.ndarray=None#crossfade计算用numpy数组
        self.f_safe_prefix_pad_length:float = 1.0
        self.input_wav:np.ndarray=None#输入音频规范化后的保存地址
        self.output_wav:np.ndarray=None#输出音频规范化后的保存地址
        self.temp_wav:np.ndarray=None#包含crossfade和输出音频的缓存区
        self.f_pitch_change:float = 0.0#float(request_form.get("fPitchChange", 0))
        self.crossfade_last:np.ndarray=None#保存上一个output的crossfade
        self.f0_mode=["parselmouth", "dio", "harvest", "crepe"]#F0预测器
        self.spk_id = 1# 默认说话人。
        self.svc_model:SvcDDSP = None
        self.launcher()#start
        # 混合说话人字典（捏音色功能）
        # 设置为非 None 字典会覆盖 spk_id
        self.spk_mix_dict = None # {1:0.5, 2:0.5} 表示1号说话人和2号说话人的音色按照0.5:0.5的比例混合
        self.use_vocoder_based_enhancer = True


    def launcher(self):
        '''窗口加载'''
        input_devices,output_devices,_, _=self.get_devices()
        sg.theme('DarkAmber')   # 设置主题
        # 界面布局
        layout = [
            [   sg.Frame(layout=[
                    [sg.Input(key='sg_model',default_text='exp\\model_chino.pt'),sg.FileBrowse('选择模型文件')]
                ],title='模型.pt格式(自动识别同目录下config.yaml)')
            ],
            [   sg.Frame(layout=[
                    [sg.Text("输入设备"),sg.Combo(input_devices,key='sg_input_device',default_value=input_devices[sd.default.device[0]])],
                    [sg.Text("输出设备"),sg.Combo(output_devices,key='sg_output_device',default_value=output_devices[sd.default.device[1]])]
                ],title='音频设备')
            ],
            [   sg.Frame(layout=[
                    [sg.Text("说话人id"),sg.Input(key='spk_id',default_text='1')],
                    [sg.Text("响应阈值"),sg.Slider(range=(-60,0),orientation='h',key='noise',resolution=1,default_value=-35)],
                    [sg.Text("变调"),sg.Slider(range=(-24,24),orientation='h',key='pitch',resolution=1,default_value=12)],
                    [sg.Text("采样率"),sg.Input(key='samplerate',default_text='44100')],
                    [sg.Checkbox(text='启用捏音色功能',default=False,key='spk_mix'),sg.Button("设置混合音色",key='set_spk_mix')]
                ],title='普通设置'),
                sg.Frame(layout=[
                    [sg.Text("音频切分大小"),sg.Slider(range=(0.1,3.0),orientation='h',key='block',resolution=0.05,default_value=0.5)],
                    [sg.Text("交叉淡化时长"),sg.Slider(range=(0.02,0.1),orientation='h',key='crossfade',resolution=0.01)],
                    [sg.Text("使用历史区块数量"),sg.Slider(range=(1,10),orientation='h',key='buffernum',resolution=1,default_value=2)],
                    [sg.Text("f0预测模式"),sg.Combo(values=self.f0_mode,key='f0_mode',default_value=self.f0_mode[2])],
                    [sg.Checkbox(text='启用增强器',default=True,key='use_enhancer')]
                ],title='性能设置'),
            ],
            [sg.Button("开始音频转换",key="start_vc"),sg.Button("停止音频转换",key="stop_vc")]
        ]

        # 创造窗口
        window = sg.Window('DDSP - GUI by INT16', layout)
        self.event_handler(window=window)


    def event_handler(self,window):
        '''事件处理'''
        while True:#事件处理循环
            event, values = window.read()
            if event ==sg.WINDOW_CLOSED:   # 如果用户关闭窗口
                self.flag_vc=False
                exit()
            if event=='start_vc' and self.flag_vc==False:
                #set values 和界面布局layout顺序一一对应
                checkpoint_path = values['sg_model']                
                self.set_devices(values["sg_input_device"],values['sg_output_device'])
                self.spk_id=int(values['spk_id'])
                threhold = values['noise']
                self.f_pitch_change = values['pitch']
                self.samplerate=int(values['samplerate'])
                block_time = float(values['block'])
                crossfade_time = values['crossfade']
                buffer_num = int(values['buffernum'])
                select_pitch_extractor=values['f0_mode']
                self.use_vocoder_based_enhancer=values['use_enhancer']
                if not values['spk_mix']:
                    self.spk_mix_dict=None
                self.block_frame=int(block_time*self.samplerate)
                self.crossfade_frame=int(crossfade_time*self.samplerate)
                self.f_safe_prefix_pad_length=block_time*(buffer_num)-crossfade_time*2
                print('crossfade_time:'+str(crossfade_time))
                print("buffer_num:"+str(buffer_num))
                print("samplerate:"+str(self.samplerate))
                print('block_time:'+str(block_time))
                print("prefix_pad_length:"+str(self.f_safe_prefix_pad_length))
                print("mix_mode:"+str(self.spk_mix_dict))
                print("enhancer:"+str(self.use_vocoder_based_enhancer))
                self.start_vc(checkpoint_path,select_pitch_extractor,threhold,buffer_num)
            if event=='stop_vc'and self.flag_vc==True:
                self.flag_vc = False
            if event=='set_spk_mix' and self.flag_vc==False:
                spk_mix = sg.popup_get_text(message='示例：1:0.3,2:0.5,3:0.2',title="设置混合音色，支持多人")
                if spk_mix != None:
                    self.spk_mix_dict=eval("{"+spk_mix.replace('，',',').replace('：',':')+"}")


    def start_vc(self,checkpoint_path,select_pitch_extractor,threhold,buffer_num):
        '''开始音频转换'''
        self.flag_vc = True
        # 是否使用预训练的基于声码器的增强器增强输出，但对硬件要求更高。
        
        enhancer_adaptive_key = 0
        # f0范围限制(Hz)
        limit_f0_min = 50
        limit_f0_max = 1100
        enable_spk_id_cover = True
        #初始化一下各个ndarray
        self.input_wav=np.zeros(int((1+buffer_num)*self.block_frame),dtype='float32')
        self.output_wav=np.zeros(self.block_frame,dtype='float32')
        self.temp_wav=np.zeros(self.block_frame+self.crossfade_frame,dtype='float32')
        self.crossfade_last=np.zeros(self.crossfade_frame,dtype='float32')
        self.fade_in_window = np.linspace(0, 1,self.crossfade_frame)
        self.fade_out_window = np.linspace(1, 0,self.crossfade_frame)
        self.svc_model = SvcDDSP(checkpoint_path, self.use_vocoder_based_enhancer, enhancer_adaptive_key, select_pitch_extractor,limit_f0_min, limit_f0_max, threhold, self.spk_id, self.spk_mix_dict, enable_spk_id_cover)
        thread_vc=threading.Thread(target=self.soundinput)
        thread_vc.start()
        
    
    def soundinput(self):
        '''
        接受音频输入
        '''
        with sd.Stream(callback=self.audio_callback, blocksize=self.block_frame,samplerate=self.samplerate,dtype='float32'):
            while self.flag_vc:
                time.sleep(self.block_time)
                print('Audio block passed.')
        print('ENDing VC')


    def audio_callback(self,indata,outdata, frames, time, status):
        '''
        音频处理
        '''
        print("Realtime VCing...")
        self.input_wav[:]=np.roll(self.input_wav,-self.block_frame)
        self.input_wav[-self.block_frame:]=librosa.to_mono(indata.T)
        print('input_wav.shape:'+str(self.input_wav.shape))
        _audio, _model_sr = self.svc_model.infer( self.f_pitch_change, self.spk_id, self.f_safe_prefix_pad_length,self.input_wav,self.samplerate)
        self.temp_wav[:] = librosa.resample(_audio, orig_sr=_model_sr, target_sr=self.samplerate)[-self.block_frame-self.crossfade_frame:]
        #cross-fade output_wav's start with last crossfade
        self.output_wav[:]=self.temp_wav[:self.block_frame]
        self.output_wav[:self.crossfade_frame]*=self.fade_in_window
        self.output_wav[:self.crossfade_frame]+=self.crossfade_last
        self.crossfade_last[:]=self.temp_wav[-self.crossfade_frame:]
        self.crossfade_last[:]*=self.fade_out_window
        print("infered _audio.shape:"+str(_audio.shape))
        outdata[:] = np.array([self.output_wav, self.output_wav]).T
        print('Outputed.')

    
    def get_devices(self,update: bool = True):
        '''获取设备列表'''
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
        output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]
        return input_devices, output_devices, input_devices_indices, output_devices_indices
    
    def set_devices(self,input_device,output_device):
        '''设置输出设备'''
        input_devices,output_devices,input_device_indices, output_device_indices=self.get_devices()
        sd.default.device[0]=input_device_indices[input_devices.index(input_device)]
        sd.default.device[1]=output_device_indices[output_devices.index(output_device)]
        print("input device:"+str(sd.default.device[0])+":"+str(input_device))
        print("output device:"+str(sd.default.device[1])+":"+str(output_device))



if __name__ == "__main__":
    gui=GUI()
