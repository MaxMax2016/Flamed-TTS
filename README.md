# Flamed-TTS: Flow Matching Attention-Free Models for Efficient Generating and Dynamic Pacing Zero-shot Text-to-Speech

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)]([https://github.com/SWivid/F5-TTS](https://github.com/flamedtts/Flamed-TTS))
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)]()
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)]()

![Overall Architecture](https://github.com/flamedtts/Flamed-TTS/blob/main/figs/Flamed-TTS.png)
<div align="center">
	<img src="https://github.com/flamedtts/Flamed-TTS/blob/main/figs/CodeDecoder_Denoiser.png" width="640" style="display: block; margin: auto;"/>
</div>

# 🔥 News
- [Coming soon] Release training instructions
- [2025.08] Release checkpoint
- [2025.08] Release inference code
- [2025.08] Init Repo
- [2025.08] Submitted to `AAAI 2026`

# 🎯 Overview

This repo implements a novel zero-shot TTS framework, named Flamed-TTS, focusing on the low-latency generation and dynamic pacing in speech synthesis.

# 🛠️ Installation Dependencies

Prepare your environment by creating a conda setup, preferably on Linux. Then, install the necessary requirements using pip:
```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n flamed-tts python=3.10
conda activate flamed-tts
cd Flamed-TTS
pip install -r requirements.txt
```

To train the model yourself, a GPU is recommended for optimal performance. However, you can generate samples using our pretrained models without requiring a GPU.

# 🚀 Inference

### Download pretrained weights

To perform inference with pretrained weights, you must download the pretrained weights for both FaCodec and OZSpeech.

* With FaCodec, you can download the FaCodec Encoder and FaCodec Decoder directly from Hugging Face: [FaCodec Encoder](https://huggingface.co/amphion/naturalspeech3_facodec/blob/main/ns3_facodec_encoder.bin), [FaCodec Decoder](https://huggingface.co/amphion/naturalspeech3_facodec/blob/main/ns3_facodec_decoder.bin).
* With Flamed-TTS, please refer [this link](https://drive.google.com/drive/folders/17A5OJoF6yUqiy62n1ghEGJ6EwHUexUEs?usp=sharing). You need to download both pretrained weights and config file for initializing model.

### Inference using python script

Script `synthesize.py` provides end-to-end pipeline for inference. Please follow the instructions:

```
python synthesize.py \
	--ckpt-path path/to/ckpt.pt \
 	--cfg-path path/to/config.yaml \
	--text "content to be synthesized" \
	--prompt-dir path/to/folder/of/prompt/audio/files \
	--prompt-list prompt_1.wav prompt_2.wav prompt_3.wav \ # list of prompt filenames to be synthesized
	--nsteps-durgen 16 \ # number of sampling steps to generate both phoneme durations and silences, 64 as default
	--nsteps-denoiser 128 \ # number of sampling steps to generate latent representations of speech, 64 as default
	--temp-durgen 1.0 \ # nosie scaling factor to generate both phoneme durations and silences, 0.3 as default
	--temp-denoiser 0.3 \ # nosie scaling factor to generate latent representations of speech, 0.3 as default
	----output-dir path/to/dir/for/output/audio/files \
	--device cuda:0 # cuda:0 as default
```

# 🔄 Training Flamed-TTS from scratch

TBD.

# ⚠️ Disclaimer

No individual or organization may use any technology described in this paper to generate, edit, or manipulate the speech of any person, including but not limited to government officials, political figures, or celebrities, without their explicit consent. Unauthorized use may violate applicable copyright, intellectual property, or privacy laws and could result in legal consequences.














