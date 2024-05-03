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
from ddspsvc.diffusion.infer_gt_mel import DiffGtMel
from ddspsvc.enhancer import Enhancer
from ast import literal_eval

app = Flask(__name__)

CORS(app)

logging.getLogger("numba").setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    raw_sample = int(float(request_form.get("sampleRate", 0)))

    # get fSafePrefixPadLength
    f_safe_prefix_pad_length = float(request_form.get("fSafePrefixPadLength", 0))
    print("f_safe_prefix_pad_length:" + str(f_safe_prefix_pad_length))
    if f_safe_prefix_pad_length > 0.025:
        silence_front = f_safe_prefix_pad_length
    else:
        silence_front = 0

    # get sample_method
    sample_method = str(request_form.get("sample_method", None))
    if sample_method == 'None':
        sample_method = 'pndm'
    else:
        sample_method = 'dpm-solver'
    print(f'sample_method:{sample_method}')

    # get speed_up
    speed_up = int(float(request_form.get("sample_interval", 20)))
    print(f'speed_up:{speed_up}')

    # get skip_steps
    skip_steps = int(float(request_form.get("skip_steps", 0)))
    print(f'skip_steps:{skip_steps}')
    kstep = 1000 - skip_steps
    if kstep < speed_up:
        kstep = 300

    # 变调信息
    key = float(request_form.get("fPitchChange", 0))

    # 获取spk_id
    raw_speak_id = str(request_form.get("sSpeakId", 0))
    print("speak_id:" + raw_speak_id)

    # http获得wav文件并转换
    input_wav_read = io.BytesIO(wave_file.read())
    # 模型推理
    _audio, _model_sr = svc_model.infer(
        input_wav=input_wav_read,
        pitch_adjust=key,
        spk_id=raw_speak_id,
        safe_prefix_pad_length=silence_front,
        acc=speed_up,
        k_step=kstep,
        method=sample_method
    )
    if raw_sample != _model_sr:
        tar_audio = librosa.resample(_audio, _model_sr, raw_sample)
    else:
        tar_audio = _audio
    # 返回音频
    out_wav_path = io.BytesIO()
    sf.write(out_wav_path, tar_audio, raw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


class SvcD3SP:
    def __init__(self, ddsp_checkpoint_path, diff_checkpoint_path, input_pitch_extractor, f0_min, f0_max):
        self.model_path = ddsp_checkpoint_path
        self.input_pitch_extractor = input_pitch_extractor
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threhold = -60
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load ddsp model
        self.model, self.args = load_model(self.model_path, device=self.device)

        # load diff_model
        self.diff_model = DiffGtMel()
        self.diff_model.flush_model(diff_checkpoint_path, self.args)

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
            device=self.device
        )

    def infer(self, input_wav, pitch_adjust, spk_id, safe_prefix_pad_length, acc, k_step, method):
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
            float(self.f0_max)
        )
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)

        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(self.threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)

        # spk_id or spk_mix_dict
        if str.isdigit(spk_id):
            spk_id = int(spk_id)
            spk_mix_dict = None
        else:
            spk_mix_dict = literal_eval(spk_id)
            spk_id = 1

        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)

        # forward and return the output
        with torch.no_grad():
            output, _, (s_h, s_n) = self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict)

            output = self.diff_model.infer(output, f0, units, volume, acc=acc, spk_id=spk_id, k_step=k_step,
                                           method=method, silence_front=silence_front,
                                           use_silence=diff_jump_silence_front, spk_mix_dict=spk_mix_dict)

            output *= mask
            output_sample_rate = self.args.data.sampling_rate

            output = output.squeeze().cpu().numpy()
            return output, output_sample_rate


if __name__ == "__main__":
    # 与冷月佬的GUI搭配使用，仓库地址:https://github.com/fishaudio/realtime-vc-gui
    # 或许还能和串佬的插件搭配(但是...已经删库了，无法测试(悲))
    # 此后端只能用于ddsp和diff都使用的浅扩散模式
    # ---------------------以下是ddsp部分的配置----------------------
    # config和模型得同一目录。
    ddsp_checkpoint_path = "exp_old/ddsp-test3/model_300000.pt"
    # ---------------------以下是diff部分的配置----------------------
    # config和模型得同一目录。
    diff_checkpoint_path = "exp_old/diffusion-test3/model_400000.pt"
    # 扩散部分完全不合成安全区，打开可以减少硬件压力并加速，但是会损失合成效果
    diff_jump_silence_front = False
    # ---------------------以下是其 他部分的配置----------------------
    # f0提取器，有 parselmouth, dio, harvest, crepe
    select_pitch_extractor = 'crepe'
    # f0范围限制(Hz)
    limit_f0_min = 50
    limit_f0_max = 1100

    svc_model = SvcD3SP(ddsp_checkpoint_path, diff_checkpoint_path, select_pitch_extractor, limit_f0_min, limit_f0_max)

    # 此处与vst插件对应，端口必须接上。
    app.run(port=6844, host="0.0.0.0", debug=False, threaded=False)
