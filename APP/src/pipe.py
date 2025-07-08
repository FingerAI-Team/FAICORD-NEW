from .audio_handler import NoiseHandler, VoiceEnhancer, AudioVisualizer
from .preprocessors import AudioFileProcessor
from .pyannotes import PyannotDIAR, PyannotVAD
from .embeddings import SBEMB, WSEMB, EMBVisualizer
from .clusters import KNNCluster
from .stt import WhisperSTT
from .llms import LLMOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from intervaltree import Interval, IntervalTree
from scipy.spatial.distance import cosine
from collections import defaultdict
from abc import abstractmethod
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import pandas as pd
import tempfile
import torch
import time 
import re 

class BasePipeline:
    def __init__(self, chunk_offset=300):
        self.set_env()    
        self.chunk_offset=chunk_offset
    
    def set_env(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

class FrontendPipe(BasePipeline):
    '''
    audio preprocess pipeline (denoise, deverve audio) 
    '''
    def __init__(self):
        super().__init__()
        self.noise_handler = NoiseHandler()
        self.audio_file_processor = AudioFileProcessor()

    def _process_chunk(self, chunk, deverve, fade_ms):
        chunk_io = BytesIO()
        chunk.export(chunk_io, format='wav')
        chunk_io.seek(0)
        denoised = self.noise_handler.denoise_audio(chunk_io)
        if deverve:
            clean_chunk = self.noise_handler.deverve_audio(denoised)
            clean_chunk.seek(0)
            seg = self.audio_file_processor.audiofile_to_AudioSeg(clean_chunk)
        else:
            seg = self.audio_file_processor.audiofile_to_AudioSeg(denoised)
        return seg

    def process_audio(self, audio_file, fade_ms=50, chunk_length=300, deverve=False):
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file)
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=chunk_length)
        print(f"[DEBUG] Chunk count: {len(chunks)}, chunk_length={chunk_length} sec")
        with ThreadPoolExecutor(max_workers=4) as executor:
            processed_chunks = list(executor.map(
                lambda c: self._process_chunk(c, deverve, fade_ms), chunks
            ))
        clean_audio = self.audio_file_processor.concat_chunk(processed_chunks)
        return clean_audio

    def save_audio(self, audio_file, file_name=None):
        self.audio_file_processor.save_audio(audio_file, file_name=file_name)


class VADPipe(BasePipeline):
    '''
    voice activity detection pipeline
    '''
    def __init__(self, config):
        super().__init__()
        self.vad_config = config 
        self.vad_model = PyannotVAD() 

    def get_vad_timestamp(self, audio_file):
        '''
        get vad timestamp of input audio 
        input: 
            - audio_file: this audio should be cleansed audio (pre-processed audio)
        output:
            - vad_timestamp 
        '''
        vad_pipeline = self.vad_model.load_pipeline_from_pretrained(self.vad_config)
        vad_timestamp = self.vad_model.get_vad_timestamp(vad_pipeline, audio_file)
        return vad_timestamp


class DIARPipe(BasePipeline):
    '''
    speaker diarization pipeline (get diar, apply vad timestamp, preprocess result) 
    '''
    def __init__(self, config):
        super().__init__()
        self.audio_file_processor = AudioFileProcessor()
        self.diar_model = PyannotDIAR()
        self.diar_config = config 
        
    def _process_chunk(self, idx, chunk, diar_pipe, num_speakers, return_embeddings):
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            chunk.export(temp_audio.name, format="wav")
            diar_result, emb = self.diar_model.get_diar_result(
                diar_pipe, temp_audio.name,
                num_speakers=num_speakers,
                return_embeddings=return_embeddings
            )
        return idx, diar_result, emb
    
    def get_diar(self, audio_file, num_speakers=None, return_embeddings=False):
        diar_pipe = self.diar_model.load_pipeline_from_pretrained(self.diar_config)
        audio_seg = self.audio_file_processor.audiofile_to_AudioSeg(audio_file) 
        chunks = self.audio_file_processor.chunk_audio(audio_seg, chunk_length=self.chunk_offset)
        results = [None] * len(chunks)
        emb_results = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=4) as executor:  # 적절히 조절
            futures = [
                executor.submit(self._process_chunk, idx, chunk, diar_pipe, num_speakers, return_embeddings)
                for idx, chunk in enumerate(chunks)
            ]
            for future in as_completed(futures):
                idx, diar_result, emb = future.result()
                results[idx] = diar_result
                emb_results[idx] = emb
        return results, emb_results

    def apply_vad(self, vad_result, diar_result):
        '''
        Re-segment diar_result using vad_result
        1. VAD에 걸친 diar segment만 resegmented_diar에 추가
        2. VAD에 걸리지 않은 diar segment는 non_overlapped_segments에 추가
        input:
            - diar_result: [[((start_time, end_time), speaker ), ... ], [((start_time, end_time), speaker), ...]]
            - vad_result: [(start_time, end_time), ...]
        output: 
            - diar result applied vad result (intersection): [((start_time, end_time), speaker), ...]
        '''
        non_overlapped_segments = []
        vad_diar = []
        vad_tree = IntervalTree(Interval(time_s, time_e) for time_s, time_e in vad_result)
        seen_segments = set()    # 중복 방지용
        for (time_s, time_e), speaker in diar_result:
            intersections = vad_tree.overlap(time_s, time_e)           
            segment_key = (round(time_s, 3), round(time_e, 3), speaker)
            if not intersections:
                non_overlapped_segments.append(((time_s, time_e), speaker))
            else:
                if segment_key not in seen_segments:
                    vad_diar.append(((time_s, time_e), speaker))
                    seen_segments.add(segment_key)
        return vad_diar

    def preprocess_result(self, diar_result, vad_result=None, emb_result=None):
        '''
        resegment, speaker_mapping 
        '''
        total_diar = self.diar_model.concat_diar_result(diar_result, chunk_offset=self.chunk_offset)
        if vad_result != None: 
            vad_diar = self.apply_vad(vad_result=vad_result, diar_result=total_diar)
            vad_diar = self.diar_model.split_diar_result(vad_diar, chunk_offset=self.chunk_offset)
            filtered_diar = self.diar_model.filter_filler(vad_diar)
            filtered_diar = self.diar_model.filter_unknown(filtered_diar)
            non_overlapped_diar = [self.diar_model.remove_overlap(diar_result) for diar_result in filtered_diar]
            return filtered_diar, non_overlapped_diar

    def save_chunk_files(self, diar_result, file_name, emb_result=None):
        '''
        save diar result, numpy emb as rttm, npy format for each chunk
        '''
        save_file_name = file_name.split('/')[-1].split('.')[0]
        for idx, chunk_diar in enumerate(diar_result):
            if len(diar_result) > 1: 
                save_file_name = f"chunk_{idx}_{file_name.split('/')[-1].split('.')[0]}"
            save_rttm_path = './dataset/rttm/' + save_file_name + '.rttm'
            self.diar_model.save_as_rttm(chunk_diar, output_rttm_path=save_rttm_path, file_name=save_file_name)
            # save_emb_path = './dataset/emb/' + save_file_name + '.npy'
            # self.diar_model.save_as_emb(emb_result[idx], output_emb_path=save_emb_path)        

    def save_merged_rttm(self, diar_result, file_name, chunk_length=300):
        """
        chunk-wise diar_result를 offset 적용해 단일 RTTM 파일로 저장
        - diar_result: List[List[((start, end), label)]], 청크별 다이얼 결과
        - save_path: 저장 경로 (.rttm)
        - file_name: rttm 내부에서 참조할 파일명
        - chunk_length: 청크 단위 시간 (초)
        """
        save_file_name = file_name.split('/')[-1].split('.')[0]
        save_rttm_path = './dataset/rttm/' + save_file_name + '.rttm'
        with open(save_rttm_path, "w") as f:
            for chunk_idx, chunk in enumerate(diar_result):
                offset = chunk_idx * chunk_length
                for (start, end), speaker in chunk:
                    abs_start = start + offset
                    duration = end - start
                    if speaker == 'filler':
                        continue
                    rttm_line = f"SPEAKER {save_file_name} 1 {abs_start:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>\n"
                    f.write(rttm_line)    

class STTPipe(BasePipeline):
    def __init__(self, whisper_api, generation_config):
        super().__init__()
        self.audio_processor = AudioFileProcessor()
        self.stt_model = WhisperSTT(whisper_api, generation_config)

    def chunk_audio(self, audio_file, chunk_length=None, start_time=None, end_time=None):
        return self.audio_processor.chunk_audio(audio_file, chunk_length, start_time, end_time)

    def prepare_audio(self, audio_file):
        return self.stt_model.prepare_whisper_audio(audio_file)

    def read_rttm(self, rttm_file):
        columns = [
            'type', 'file_id', 'channel', 'start', 'duration',
            'ortho', 'stype', 'speaker', 'conf', 'slat'
        ]
        df = pd.read_csv(rttm_file, sep=' ', header=None, names=columns, engine='python')
        df = df[['file_id', 'start', 'duration', 'speaker']]
        return df

    def transcribe_by_rttm(self, whisper_audio, diar_result, transcribe_type='api'):
        results = []
        text_filter = dict()
        text_filter['temperature'] = 0.8
        text_filter['no_speech_prob'] = 0.5
        if transcribe_type == 'api' and diar_result is not None:
            waveform, sample_rate = self.stt_model.prepare_whisper_audio(whisper_audio)
            print(f'sample_rate: {sample_rate}')
            # waveform, sample_rate = whisper_audio  # waveform: Tensor (1, N)
            for idx, row in diar_result.iterrows():
                start_sec = row['start']
                end_sec = row['start'] + row['duration']
                speaker = row['speaker']

                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]
                stt_result = self.stt_model.transcribe_text_api((segment_waveform, sample_rate))
                # print(len(stt_result))
                if stt_result != None:
                    text_result = self.stt_model.extract_text(stt_result, text_filter)
                    # print(text_result) 
                results.append({
                    'speaker': speaker,
                    'text': text_result
                })
        return results

    def extract_only_text(self, segments):
        '''
        input: segments 
        output: text
        '''
        texts = "" 
        for seg in segments: 
            texts += seg.text + " "
        return texts

    def merge_consecutive_same_speaker(self, df):
        merged = []
        df = df.sort_values('start').reset_index(drop=True)
        cur_start = df.loc[0, 'start']
        cur_end = cur_start + df.loc[0, 'duration']
        cur_speaker = df.loc[0, 'speaker']
        file_id = df.loc[0, 'file_id']
        for i in range(1, len(df)):
            row = df.loc[i]
            start = row['start']
            end = start + row['duration']
            speaker = row['speaker']
            if speaker == cur_speaker:
                # 연속된 동일 화자면 확장
                cur_end = end
            else:
                # 다른 화자면 지금까지 병합한 것 저장
                merged.append([file_id, round(cur_start, 6), round(cur_end - cur_start, 6), cur_speaker])
                # 다음 화자로 초기화
                cur_start = start
                cur_end = end
                cur_speaker = speaker
        # 마지막 화자 블록 저장
        merged.append([
            file_id, round(cur_start, 6), round(cur_end - cur_start, 6), cur_speaker
        ])
        return pd.DataFrame(merged, columns=[
            'file_id', 'start', 'duration', 'speaker'
        ])


class SummaryPipe(BasePipeline):
    '''
    회의록 요약 파이프라인
    '''
    def __init__(self, config, api_key):
        super().__init__()
        self.config = config
        self.api_key = api_key

    def set_openai_client(self):
        openai_summary_model = LLMOpenAI(config=self.config, api_key=self.api_key)
        return openai_summary_model

    def summarize(self, summary_model, text):
        '''
        회의록 요약
        input:
            - text: 회의록 텍스트
            - file_name: 파일 이름 (선택적)
        output:
            - summary: 요약된 텍스트
        '''
        summary_model.set_generation_config()
        summary_model.set_summary_guideline()
        print('', end='\n\n')
        print(summary_model.system_role, end='\n\n')
        prompt_template = summary_model.set_prompt_template(text)
        return summary_model.get_response(prompt_template, role=summary_model.system_role) #, sub_role=summary_model.sub_role)        
        

class PostProcessPipe(BasePipeline):
    '''
    filler 처리된 diar result와, non-overlapped diar 후처리하는 클래스. 
    1. knn clustering을 이용해 non-overlapped diar 내 화자 리레이블링 -> relabeled non overlapped diar  (func. relabel_nonoverlapped_labels)
    2. relabeled non-overlapped diar에서 화자별 임베딩 계산 (func. get_chunk_emb_array)
    3. relabeled non-overlapped diar을 이용해 계산한 청크별 화자 고유 임베딩 값 -> 청크별 화자 매핑 딕셔너리 생성 (func. build_label_mapping_dict)
    4. relabeled non-overlapped diar을 이용한 full diar re-labeling (func. apply labels to full diar)
    5. 청크별 화자 매핑 딕셔너리 -> full diar re-labeled 결과에 적용 (func. apply_label_mapping_to_diar)
    '''
    def __init__(self, chunk_offset=300):
        super().__init__(chunk_offset)
        self.wsemb = WSEMB()
        self.emb_model = self.wsemb.load_model(model_path='./pretrained_models/voxceleb_resnet221_LM')
        self.knn_cluster = KNNCluster() 
        self.emb_visualizer = EMBVisualizer()
              
    def get_chunk_emb_array(self, file_name, diar_result):
        '''
        get speaker emb array for each audio chunk 
        input:
            - file_name: audio file name 
            - diar result: diar results of audio chunk, each diar result is consists of [[((start, end), speaker), ((start, end), speaker), ...], [(())]]
        output:
            - chunk emb array: (chunk_idx, emb_array, original_labels, segment_bounds)
                - segment_bounds: (start, end)
        '''
        chunk_emb_array = []
        for idx, diar in enumerate(diar_result):
            emb_result = self.wsemb.get_embeddings_from_diar(
                self.emb_model, file_name, diar, chunk_offset=idx*self.chunk_offset
            )
            emb_array = np.vstack([emb for (_, _, emb) in emb_result])
            original_labels = [speaker for (_, speaker, _) in emb_result]
            segments = [(start, end) for ((start, end), _, _) in emb_result]
            chunk_emb_array.append((idx, emb_array, original_labels, segments))
        return chunk_emb_array    

    def build_label_mapping_dict(self, chunk_emb_array, threshold=0.6):
        '''
        Maps speakers across chunks using speaker embeddings.
        input:
            - chunk_emb_array: List of tuples like (chunk_idx, emb_array, original_labels, segment_bounds)
            - threshold: Similarity threshold for matching speakers
        output:
            - chunkwise_mapping: Dict of chunk_idx → local_to_global speaker label mapping
        '''
        speaker_registry = {}  # global_label: centroid
        chunkwise_mapping = {}
        def get_next_speaker_name():
            existing_ids = [
                int(re.search(r'\d+', label).group())
                for label in speaker_registry.keys()
                if re.match(r'^SPEAKER_\d+$', label)
            ]
            next_id = max(existing_ids) + 1 if existing_ids else 0
            return f'SPEAKER_{next_id:02d}'

        for chunk_idx, emb_array, original_labels, segment_bounds in chunk_emb_array:
            # print(f"[DEBUG] chunk {chunk_idx} — labels: {original_labels}")
            speaker_to_embs = defaultdict(list)
            for emb, label in zip(emb_array, original_labels):
                if label == 'UNKNOWN':
                    print('detected unknown')
                    continue
                speaker_to_embs[label].append(emb)
            speaker_centroids = {
                speaker: np.mean(np.stack(embs), axis=0)
                for speaker, embs in speaker_to_embs.items()
            }
            mapping = {}
            current_chunk_registered = {}  # 현재 청크 내에서 방금 등록한 speaker
            for speaker, centroid in speaker_centroids.items():
                if chunk_idx == 0:
                    speaker_registry[speaker] = centroid
                    mapping[speaker] = speaker
                    print(f"[INIT][chunk {chunk_idx}] {speaker} → {speaker}")
                else:
                    best_similarity = -1
                    best_key = None
                    # global registry 비교
                    for reg_label, reg_centroid in speaker_registry.items():
                        similarity = self.calc_emb_similarity(torch.tensor(reg_centroid), torch.tensor(centroid))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_key = reg_label
                    # 같은 chunk 내에서 방금 등록한 speaker들과도 비교
                    for reg_label, reg_centroid in current_chunk_registered.items():
                        similarity = self.calc_emb_similarity(torch.tensor(reg_centroid), torch.tensor(centroid))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_key = reg_label
                    if best_similarity >= threshold:
                        mapping[speaker] = best_key
                        print(f"[MAP][chunk {chunk_idx}] {speaker} → {best_key} (sim={best_similarity:.2f})")
                    else:
                        new_label = get_next_speaker_name()
                        speaker_registry[new_label] = centroid
                        current_chunk_registered[new_label] = centroid
                        mapping[speaker] = new_label
                        print(f"[NEW][chunk {chunk_idx}] {speaker} → {new_label} 등록됨 (sim={best_similarity:.2f})")
            chunkwise_mapping[chunk_idx] = mapping
        return chunkwise_mapping

    def relabel_nonoverlapped_labels(self, file_name, diar_result, k=5):
        '''
        input:
            - non overlapped diar result: to get speaker emb
        return:
            - relabeled non overlapped diar result: apply knn clustering 
        '''
        relabeled_diar_result = []
        chunk_emb_data = self.get_chunk_emb_array(file_name, diar_result)
        for (idx, emb_array, original_labels, segments) in chunk_emb_data:
            new_labels = self.knn_cluster.relabel_by_knn(emb_array, original_labels, k=k)
            emb_segment_set = set(segments)
            emb_idx = 0
            relabeled_diar = []
            for segment in diar_result[idx]:
                (start, end), original_label = segment
                if original_label == 'filler':
                    relabeled_diar.append(segment)
                elif original_label == 'UNKNOWN':
                    relabeled_diar.append(segment)
                elif (start, end) in emb_segment_set:
                    new_label = new_labels[emb_idx]
                    relabeled_diar.append(((start, end), new_label))
                    emb_idx += 1
                else:
                    relabeled_diar.append(segment)
            relabeled_diar_result.append(relabeled_diar)
        return relabeled_diar_result  

    def apply_labels_to_full_diar(self, full_diar, relabeled_nonoverlap_diar, min_ratio=0.3):
        '''
        full_diar[0]: chunk 0 diar    - [((start, end), speaker), ((start, end), speaker), ... ]  
        full_diar[1]: chunk 1 diar    -                         '' 
        '''
        relabeled_full_diar = []
        for idx, chunk in enumerate(full_diar):
            chunk_diar = []
            for (full_start, full_end), full_label in chunk:
                overlap_segments = []
                earliest_speaker = None
                earliest_start = float('inf')
                speaker_durations = {}  # 각 speaker의 전체 발화 시간 저장
                for (rel_start, rel_end), rel_label in relabeled_nonoverlap_diar[idx]:
                    # 겹치는 경우만 고려
                    overlap_start = max(full_start, rel_start)
                    overlap_end = min(full_end, rel_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    if overlap_duration > 0:
                        overlap_segments.append((overlap_duration, rel_label))
                        if rel_label not in speaker_durations:
                            speaker_durations[rel_label] = rel_end - rel_start  # 발화 길이 저장
                        if rel_start < earliest_start:
                            earliest_start = rel_start
                            earliest_speaker = rel_label
                if earliest_speaker is not None:   # 가장 많이 겹친 사람 찾기
                    early_duration = speaker_durations.get(earliest_speaker, 0)
                    dominant_label = max(overlap_segments, key=lambda x: x[0])[1]
                    dominant_duration = speaker_durations.get(dominant_label, 0)
                    if dominant_label != earliest_speaker and early_duration < dominant_duration * min_ratio:
                        best_label = dominant_label
                    else:
                        best_label = earliest_speaker
                else:
                    best_label = full_label  # 겹치는 발화가 없으면 fallback
                chunk_diar.append(((full_start, full_end), best_label))
            relabeled_full_diar.append(chunk_diar)
        return relabeled_full_diar

    def calc_emb_similarity(self, emb1, emb2, model_type='sb'):
        if model_type == 'sb':   # [1, 1, 192] 
            emb1 = emb1.view(-1).cpu().numpy()
            emb2 = emb2.view(-1).cpu().numpy()
            similarity = 1 - cosine(emb1, emb2)
            return similarity
        elif model_type == 'wespeaker':
            emb1 = emb1.view(-1).cpu().numpy()
            emb2 = emb2.view(-1).cpu().numpy()            
            return 1 - cosine(emb1, emb2)    # cosine()은 distance니까 1 - distance

    def apply_label_mapping_to_diar(self, diar_results, chunkwise_mapping):
        relabeled = []
        for chunk_idx, diar in enumerate(diar_results):
            mapping = chunkwise_mapping.get(chunk_idx, {})
            chunk_result = []
            for (start, end), label in diar:
                if label in ['filler', 'UNKNOWN']:
                    chunk_result.append(((start, end), label))
                else:
                    new_label = mapping.get(label, label)
                    chunk_result.append(((start, end), new_label))
            relabeled.append(chunk_result)    # 청크별 결과를 이중 리스트로 유지
        return relabeled