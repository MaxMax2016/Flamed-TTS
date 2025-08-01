import os
import torch
import argparse
from oz2 import OZ2
from tqdm import tqdm
import soundfile as sf
from omegaconf import OmegaConf
from oz2.models.facodec import FACodecEncoder, FACodecDecoder


SR = 16000
CURDIR = os.path.dirname(__file__)


def get_codec(device):
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = os.path.join(CURDIR, 'oz2', 'models', 'facodec', 'checkpoints', 'ns3_facodec_encoder.bin')
    decoder_ckpt = os.path.join(CURDIR, 'oz2', 'models', 'facodec', 'checkpoints', 'ns3_facodec_decoder.bin')
    fa_encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
    fa_encoder.eval()
    fa_decoder.eval()
    
    return fa_encoder, fa_decoder


def synthesize(args):
    text_file = args.text_file
    input_dir = args.input_dir
    output_dir = args.output_dir
    ckpt_path = args.ckpt_path
    cfg_path = args.cfg_path
    weights_only = args.weights_only
    nsteps_durgen = args.nsteps_durgen
    nsteps_denoiser = args.nsteps_denoiser
    temp_durgen = args.temp_durgen
    temp_denoiser = args.temp_denoiser
    device = args.device
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(os.path.join(output_dir, f'nfe{nsteps_denoiser}-temp{temp_denoiser}')):
        os.mkdir(os.path.join(output_dir, f'nfe{nsteps_denoiser}-temp{temp_denoiser}'))
    
    cfg = OmegaConf.load(cfg_path)
    codec_encoder, codec_decoder = get_codec(device)
    model = OZ2.from_pretrained(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
        weights_only=weights_only,
        training_mode=False
    )

    infer_times, output_durations = [], []
    with open(text_file, 'r') as fin:
        for line in tqdm(list(fin)):
            filename, prompt_filename, transcript, _, _, dur  = line.rstrip().split('|')
            prompt_filepath = os.path.join(input_dir, prompt_filename)

            if os.path.exists(os.path.join(output_dir, f'nfe{nsteps_denoiser}-temp{temp_denoiser}', f'{filename}')):
                continue

            results = model.sample(
                text=transcript,
                prompt_raw=prompt_filepath,
                sr=SR,
                codec_encoder=codec_encoder,
                codec_decoder=codec_decoder,
                nsteps_durgen=nsteps_durgen,
                nsteps_denoiser=nsteps_denoiser,
                temp_durgen=temp_durgen,
                temp_denoiser=temp_denoiser,
            )
    
            wav = results['wav']
            infer_times.append(results['time'])
            output_durations.append(len(wav) / SR)

            sf.write(
                file=os.path.join(output_dir, f'nfe{nsteps_denoiser}-temp{temp_denoiser}', f'{filename}'), 
                data=wav,
                samplerate=SR
            )
    
    rtf = [t/d for t, d in zip(infer_times, output_durations)]
    rtf_mean = sum(rtf) / len(rtf)
    return rtf_mean


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-file', type=str, required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--cfg-path', type=str, required=True)
    parser.add_argument('--weights-only', type=str, default=True)
    parser.add_argument('--temp-durgen', type=float, default=0.3)
    parser.add_argument('--temp-denoiser', type=float, default=0.3)
    parser.add_argument('--nsteps-durgen', type=int, default=8)
    parser.add_argument('--nsteps-denoiser', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    rtf = synthesize(args)
    
    print('='*20, 'Avg RTF', '='*20)
    print('>'*5, f'RTF: ', round(rtf, 3))