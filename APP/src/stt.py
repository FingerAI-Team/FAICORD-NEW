from abc import abstractmethod
from pydub import AudioSegment
from openai import OpenAI
from datetime import timedelta
import tempfile
import base64
import torch
import json 
import os
import io 

class STTModule:
    def __init__(self, openai_api=None):
        self.openai_api = openai_api

    @abstractmethod
    def set_client(self):
        pass


class WhisperSTT(STTModule):
    def __init__(self, openai_api, generation_config):
        self.load_word_dictionary(os.path.join('./config', 'word_dict.json')) 
        self.set_client(openai_api)
        self.generation_config = generation_config 

    def set_client(self, openai_api):
        self.openai_client = OpenAI(api_key=openai_api)
    
    def format_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        return str(td)[:-3].zfill(8)

    def load_word_dictionary(self, word_dict_path):
        with open(word_dict_path, mode='r', encoding='utf-8') as file:
            self.word_dict = json.load(file)    # JSON 데이터를 한번만 로드

    def apply_word_dictionary(self, stt_text, word_dict):
        for incorrect_word, correct_word in word_dict.items():
            stt_text = stt_text.replace(incorrect_word, correct_word)
        return stt_text

    def prepare_whisper_audio(self, audio_input, sample_rate=16000):
        if isinstance(audio_input, AudioSegment):
            audio = audio_input
        elif isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)
            audio = AudioSegment.from_file(audio_input, format="wav")
        elif isinstance(audio_input, str):  # 파일 경로
            audio = AudioSegment.from_file(audio_input)
        else:
            raise TypeError("지원되지 않는 오디오 타입입니다.")
        
        # Whisper-friendly audio: 16kHz, mono, 2byte
        audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        samples = audio.get_array_of_samples()
        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0) / 32768.0  # normalize to [-1, 1]
        return waveform, sample_rate

    def transcribe_text_api(self, audio_file_or_tensor):
        '''
        transcription.segments: segment.start, segment.end, segment.text, ...
        '''
        if isinstance(audio_file_or_tensor, tuple) and isinstance(audio_file_or_tensor[0], torch.Tensor):
            waveform, sample_rate = audio_file_or_tensor
            # waveform: Tensor (1, N), float32
            # → numpy array → bytes → AudioSegment
            array = (waveform.squeeze().numpy() * 32767).astype("int16")
            audio_segment = AudioSegment(
                array.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
        else:
            audio_segment = self.prepare_whisper_audio(audio_file_or_tensor)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_segment.export(temp_audio_file.name, format="wav")
            with open(temp_audio_file.name, "rb") as f:
                #try:
                    transcription = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language='ko',
                        response_format="verbose_json",
                    )
                #except: 
                #    print(f'audio transcript err')
            os.remove(temp_audio_file.name)
        try:
            return transcription.segments
        except:
            print(f'err occured')
            return None 

    def extract_text(self, segments, text_filter=None):
        '''
        filter['temperature'] = 1.0
        filter['no_speech_prob'] = 1.0
        '''
        if len(segments) == 1: 
            segment = segments[0]
            if segment.temperature < text_filter['temperature'] and segment.no_speech_prob < text_filter['no_speech_prob']:   
                segment.text = segment.text.strip()
                segment.text = self.apply_word_dictionary(segment.text, self.word_dict)
            return segment.text 
        else:
            text = "" 
            for segment in segments: 
                if segment.temperature < text_filter['temperature'] and segment.no_speech_prob < text_filter['no_speech_prob']:   
                    segment.text = segment.text.strip()
                    segment.text = self.apply_word_dictionary(segment.text, self.word_dict)
                    text = text + segment.text + " "
            return text

    def save_as_txt(self, segments, file_name):
        '''
        file_name
        '''
        lines = []
        for seg in segments:
            start = self.format_timestamp(seg.start)
            end = self.format_timestamp(seg.end)
            text = seg.text
            temp = seg.temperature
            no_speech = round(seg.no_speech_prob, 2)
            line = f"[{start}-{end}, {no_speech}] {text}"
            lines.append(line)
        
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))