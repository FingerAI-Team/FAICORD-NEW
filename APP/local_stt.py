
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio

# model_id = "ghost613/whisper-large-v3-turbo-korean"
# WhisperProcessor.from_pretrained(model_id).save_pretrained("./pretrained_models/whisper_local")
# WhisperForConditionalGeneration.from_pretrained(model_id).save_pretrained("./pretrained_models/whisper_local")    

processor = WhisperProcessor.from_pretrained("./pretrained_models/whisper_local")
model = WhisperForConditionalGeneration.from_pretrained("./pretrained_models/whisper_local").to("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(), target_sr

# 슬라이딩 윈도우 단위로 자르기 및 인식
def transcribe_long_audio(path, chunk_length_sec=30, stride_sec=5):
    waveform, sr = load_audio(path)
    chunk_size = chunk_length_sec * sr
    stride_size = stride_sec * sr
    result = []
    for start in range(0, waveform.size(0), chunk_size - stride_size):
        end = min(start + chunk_size, waveform.size(0))
        chunk = waveform[start:end]
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(inputs["input_features"].to(model.device))
            text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            result.append(text)
        if end == waveform.size(0):
            break
    return " ".join(result)

# 사용 예시
audio_path = "./dataset/audio/test_20250226.wav"
full_text = transcribe_long_audio(audio_path)
print("📝 전체 인식 결과:\n", full_text)
