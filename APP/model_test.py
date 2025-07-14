from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("talkbank/callhome", "eng", split="data")

# 이제 각 example은 dict입니다.
for example in ds:
    audio_info = example["audio"]
    audio_array = audio_info["array"]
    sampling_rate = audio_info["sampling_rate"]
    
    print(f"wave shape: {audio_array.shape}, rate: {sampling_rate}")