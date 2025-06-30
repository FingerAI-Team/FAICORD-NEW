from .audio_handler import NoiseHandler, VoiceEnhancer, AudioVisualizer
from .preprocessors import AudioFileProcessor
from .pyannotes import PyannotVAD
from .stt import WhisperSTT
from scipy.spatial.distance import cosine
from abc import abstractmethod
from pydub import AudioSegment
from typing import List, Tuple
from io import BytesIO
import tempfile


class BasePipeline:
    @abstractmethod
    def set_env(self):
        pass


class FrontendPipe(BasePipeline):
    def set_env(self):
        self.noise_handler = NoiseHandler()
        self.voice_enhancer = VoiceEnhancer()
        self.audio_file_processor = AudioFileProcessor()

    def process_audio(self, audio_file, fade_ms=50, chunk_length=300, deverve=False):
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file) 
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=chunk_length)
        print(f"[DEBUG] Chunk count: {len(chunks)}, chunk_length={chunk_length} sec")
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_io = BytesIO()
            chunk.export(chunk_io, format='wav')
            chunk_io.seek(0)
            denoised = self.noise_handler.denoise_audio(chunk_io)
            print(f"[DEBUG] Denoised chunk duration: {len(chunk) / 1000} sec")
            if deverve == True:             
                clean_chunk = self.noise_handler.deverve_audio(denoised)
                clean_chunk.seek(0)
                seg = self.audio_file_processor.audiofile_to_AudioSeg(clean_chunk)
                # seg = seg.fade_in(fade_ms).fade_out(fade_ms)
            else:
                seg = self.audio_file_processor.audiofile_to_AudioSeg(denoised)
                # seg = seg.fade_in(fade_ms).fade_out(fade_ms)
            processed_chunks.append(seg)
        clean_audio = self.audio_file_processor.concat_chunk(processed_chunks)
        # normalized_audio = self.voice_enhancer.normalize_audio_lufs(clean_audio)
        return clean_audio

    def save_audio(self, audio_file, file_name=None):
        self.audio_file_processor.save_audio(audio_file, file_name=file_name)


class VADPipe(BasePipeline):
    def set_env(self, vad_config):
        self.vad_model = PyannotVAD()
        self.vad_config = vad_config
        self.audio_visualizer = AudioVisualizer()

    def get_vad_timestamp(self, audio_file):
        vad_pipeline = self.vad_model.load_pipeline_from_pretrained(self.vad_config)
        vad_timestamp = self.vad_model.get_vad_timestamp(vad_pipeline, audio_file)
        return vad_timestamp

    def merge_segments(
        self,
        vad_segments,
        min_length=5,
        silence_gap=5,
        min_keep_length=0.5
    ):
        if not vad_segments:
            return []
        merged = []
        idx = 0
        while idx < len(vad_segments):
            start, end = vad_segments[idx]
            duration = end - start
            if duration >= min_length:
                merged.append((start, end))
                idx += 1
                continue
            acc_start, acc_end, acc_idx = start, end, idx 
            acc_duration = duration

            # 다음 세그먼트들과 이어붙이기
            while acc_duration < min_length and acc_idx + 1 < len(vad_segments):
                next_start, next_end = vad_segments[acc_idx + 1]
                gap = next_start - acc_end
                if gap >= silence_gap:
                    break    # 침묵 → 병합 종료
                acc_end = next_end
                acc_duration = acc_end - acc_start
                acc_idx += 1
            if acc_duration >= min_length:
                merged.append((acc_start, acc_end))
                idx = acc_idx + 1     # 병합된 다음 세그먼트로 이동
            else:
                if acc_end - acc_start >= min_keep_length:
                    merged.append((acc_start, acc_end))
                idx = acc_idx + 1
        return merged


class STTPipe(BasePipeline):
    def set_env(self, whisper_api, generation_config):
        self.audio_processor = AudioFileProcessor()
        self.stt_model = WhisperSTT(whisper_api, generation_config)

    def chunk_audio(self, audio_file, chunk_length=None, start_time=None, end_time=None):
        return self.audio_processor.chunk_audio(audio_file, chunk_length, start_time, end_time)

    def transcribe_text(self, audio_file, vad_result=None, chunk_length=270, transcribe_type='api'):
        '''

        '''
        whisper_audio = self.stt_model.prepare_whisper_audio(audio_file)
        results = []
        if transcribe_type == 'api' and vad_result == None: 
            audio_chunk = self.chunk_audio(whisper_audio, chunk_length=chunk_length)
            for idx, chunk in enumerate(audio_chunk):
                stt_result = self.stt_model.transcribe_text_api(chunk)
                results.extend(stt_result) 
            return results 
        elif transcribe_type == 'api' and vad_result != None:
            for time_s, time_e in vad_result:    # (start, end)
                audio_chunk = self.chunk_audio(whisper_audio, start_time=time_s, end_time=time_e)[0]
                try:
                    stt_result = self.stt_model.transcribe_text_api(audio_chunk)
                except:
                    print(f'audio too short: {time_s}~{time_e}')
                results.extend(stt_result)
            return results
        elif transcribe_type == 'local':
            pass 

    def extract_only_text(self, segments):
        '''
        input: segments 
        output: text
        '''
        texts = "" 
        for seg in segments: 
            texts += seg.text + " "
        return texts 

    def postprocess_result(self, stt_result, no_speech_prob=0.9, temperature=0.5, file_name=None):
        '''
        extract text from segments  + apply word dictionary 
        '''
        text_filter = {
            'no_speech_prob': no_speech_prob,
            'temperature': temperature
        }
        for seg in stt_result: 
            seg.text = self.stt_model.extract_text(seg, text_filter=text_filter)

        if file_name != None: 
            self.stt_model.save_as_txt(stt_result, file_name)
        return stt_result 


class PostProcessPipe(BasePipeline):
    def set_env(self, emb_config):
        self.audio_file_processor = AudioFileProcessor()
        self.emb_model = SBEMB(emb_config)

    def get_vectoremb(self):
        pass 

    def map_speaker(self):
        pass