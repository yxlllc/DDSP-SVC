import io
import logging
import torch
import numpy as np
from . import slicerimport soundfile as sf
import librosa
from flask import Flask, request, send_file
from flask_cors import CORS

from ddspsvc.ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddspsvc.ddsp.core import upsample
from ddspsvc.enhancer import Enhancer


app = Flask(__name__)

CORS(app)

logging.getLogger("numba").setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # get fSafePrefixPadLength
    f_safe_prefix_pad_length = float(request_form.get("fSafePrefixPadLength", 0))
    print("f_safe_prefix_pad_length:"+str(f_safe_prefix_pad_length))
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # 获取spk_id
    int_speak_id = int(request_form.get("sSpeakId", 0))
    if enable_spk_id_cover:
        int_speak_id = spk_id
    # print("说话人:" + str(int_speak_id))
    # DAW所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    # http获得wav文件并转换
    input_wav_read = io.BytesIO(wave_file.read())
    # 模型推理
    _audio, _model_sr = svc_model.infer(input_wav_read, f_pitch_change, int_speak_id, f_safe_prefix_pad_length)
    tar_audio = librosa.resample(_audio, _model_sr, daw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    sf.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


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
        if self.args.data.encoder == 'cnhubertsoftfish':
            cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=self.device)
        
        # load enhancer
        if self.vocoder_based_enhancer:
            self.enhancer = Enhancer(self.args.enhancer.type, self.args.enhancer.ckpt, device=self.device)

    def infer(self, input_wav, pitch_adjust, speaker_id, safe_prefix_pad_length):
        print("Infer!")
        # load input
        audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
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


if __name__ == "__main__":
    # ddsp-svc下只需传入下列参数。
    # 对接的是串串香火锅大佬https://github.com/zhaohui8969/VST_NetProcess-。建议使用最新版本。
    # flask部分来自diffsvc小狼大佬编写的代码。
    # config和模型得同一目录。
    checkpoint_path = "exp/multi_speaker/model_300000.pt"
    # 是否使用预训练的基于声码器的增强器增强输出，但对硬件要求更高。
    use_vocoder_based_enhancer = True
    # 结合增强器使用，0为正常音域范围（最高G5)内的高音频质量，大于0则可以防止超高音破音
    enhancer_adaptive_key = 0
    # f0提取器，有parselmouth, dio, harvest, crepe
    select_pitch_extractor = 'crepe'
    # f0范围限制(Hz)
    limit_f0_min = 50
    limit_f0_max = 1100
    # 音量响应阈值(dB)
    threhold = -60
    # 默认说话人。以及是否优先使用默认说话人覆盖vst传入的参数。
    spk_id = 1
    enable_spk_id_cover = True
    # 混合说话人字典（捏音色功能）
    # 设置为非 None 字典会覆盖 spk_id
    spk_mix_dict = None # {1:0.5, 2:0.5} 表示1号说话人和2号说话人的音色按照0.5:0.5的比例混合
    svc_model = SvcDDSP(checkpoint_path, use_vocoder_based_enhancer, enhancer_adaptive_key, select_pitch_extractor,
                        limit_f0_min, limit_f0_max, threhold, spk_id, spk_mix_dict, enable_spk_id_cover)

    # 此处与vst插件对应，端口必须接上。
    app.run(port=6844, host="0.0.0.0", debug=False, threaded=False)
