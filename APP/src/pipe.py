from .audio_handler import NoiseHandler, VoiceEnhancer, AudioVisualizer
from .preprocessors import AudioFileProcessor
from .pyannotes import PyannotDIAR, PyannotVAD
from .embeddings import SBEMB, WSEMB, EMBVisualizer
from .clusters import KNNCluster
from .stt import WhisperSTT
from .llms import LLMOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cosine
from collections import defaultdict
from abc import abstractmethod
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import torch
import json
import re 


class BasePipeline:
    def __init__(self, chunk_offset=300):
        self.set_env()    
        self.chunk_offset=chunk_offset
    
    def set_env(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

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
    
    def transcribe_by_rttm(self, whisper_audio, diar_result, transcribe_type='api', max_workers=8):
        results = []
        text_filter = {'temperature': 0.8, 'no_speech_prob': 0.5}
        if transcribe_type == 'api' and diar_result is not None:
            waveform, sample_rate = self.stt_model.prepare_whisper_audio(whisper_audio)
            segments = []
            for idx, row in diar_result.iterrows():
                start_sec = row['start']
                end_sec = row['start'] + row['duration']
                speaker = row['speaker']

                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]
                segments.append((segment_waveform, sample_rate, speaker, start_sec))

            def transcribe_segment_safe(segment_waveform, sample_rate, speaker, start_sec, retry=3):
                for attempt in range(retry):
                    try:
                        stt_result = self.stt_model.transcribe_text_api((segment_waveform, sample_rate))
                        if stt_result:
                            text_result = self.stt_model.extract_text(stt_result, text_filter)
                            return {'speaker': speaker, 'text': text_result, 'start': start_sec}
                    except Exception as e:
                        print(f"[Retry {attempt+1}] Error for speaker {speaker}: {e}")
                        time.sleep(1)
                return {'speaker': speaker, 'text': None, 'start': start_sec}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(transcribe_segment_safe, seg[0], seg[1], seg[2], seg[3])
                    for seg in segments
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error during transcription: {e}")
            results.sort(key=lambda x: x['start'])
            for r in results:
                del r['start']
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
    
    def read_stt_result(self, stt_path):
        with open(stt_path, "r", encoding="utf-8") as f:
            stt_result = json.load(f)
        return stt_result

    def split_stt_result(self, stt_result, chunk_count=3):
        total_len = len(stt_result)
        chunk_size = total_len // chunk_count
        chunks = []
        for i in range(chunk_count):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < chunk_count - 1 else total_len
            chunks.append(stt_result[start:end])
        return chunks  # [초반부, 중반부, 후반부]

    def summarize(self, summary_model, text, system_prompt=None, subrole_prompt=None):
        '''
        회의록 요약
        input:
            - text: 회의록 텍스트
            - file_name: 파일 이름 (선택적)
        output:
            - summary: 요약된 텍스트
        '''
        summary_model.set_generation_config()
        summary_model.set_summary_guideline(system_prompt, subrole_prompt)
        # print('', end='\n\n')
        # print(summary_model.system_role, end='\n\n')
        prompt_template = summary_model.set_prompt_template(text)
        return summary_model.get_response(prompt_template, role=summary_model.system_role, sub_role=summary_model.sub_role)        

    def convert_minutes_to_markdown(self, raw_text: str) -> str:
        lines = raw_text.strip().split('\n')
        md_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                md_lines.append('')
            elif line.startswith('## '):   # 예: ## 안건
                md_lines.append(f'# {line[3:].strip()}')
            elif line.startswith('### '):   # 예: ### 안건 1:
                md_lines.append(f'## {line[4:].strip()}')
            elif line.startswith('- '):   # Bullet point
                md_lines.append(f'- {line[2:].strip()}')
            elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.'):
                md_lines.append(f'{line.strip()}')   # 번호 붙은 항목은 그대로
            elif line.startswith('**') and '**' in line[2:]:   # 발언자 라인
                md_lines.append(f'- {line}')
            else:
                md_lines.append(line)
        return '\n'.join(md_lines)    

    def format_stt_to_prompt(self, stt_list):
        """
        STT 결과 리스트를 학습용 prompt 텍스트로 변환
        """
        lines = []
        for item in stt_list:
            speaker = item.get("speaker", "")
            text = item.get("text", "").strip()
            if speaker and text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def build_jsonl_entry(self, stt_list, gpt_summary):
        """
        단일 jsonl 데이터 구성 (prompt + response)
        """
        prompt = self.format_stt_to_prompt(stt_list)
        return {
            "prompt": prompt,
            "response": gpt_summary.strip()
        }

    def export_jsonl(self, data_pairs, output_path):
        """
        data_pairs: list of (stt_list, gpt_summary) tuples
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for stt, summary in data_pairs:
                entry = self.build_jsonl_entry(stt, summary)
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"[✔] Saved {len(data_pairs)} samples to {output_path}")


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

    def _l2norm(self, v, eps=1e-12):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v if n == 0 else v / (n + eps)

    def _centroid_norm(self, vecs):
        V = np.stack([self._l2norm(v) for v in vecs], axis=0)  # 세그 임베딩 정규화 보장
        return self._l2norm(V.mean(axis=0))

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
        """
        Maps speakers across chunks using (L2-normalized) speaker centroids.
        - many-to-one 매핑 허용
        - 임베딩/센트로이드 모두 L2 정규화 → dot == cosine
        """
        import re
        from collections import defaultdict

        speaker_registry = {}  # global_label -> centroid (L2-normalized)
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
            # 라벨별 세그 임베딩 수집 (세그 임베딩도 L2 정규화해서 평균 안정화)
            speaker_to_embs = defaultdict(list)
            for emb, label in zip(emb_array, original_labels):
                if label == 'UNKNOWN':
                    continue
                speaker_to_embs[label].append(self._l2norm(emb))

            # 라벨별 센트로이드 계산 후 L2 정규화
            speaker_centroids = {
                spk: self._l2norm(np.mean(np.stack(embs, axis=0), axis=0))
                for spk, embs in speaker_to_embs.items()
            }
            mapping = {}
            current_chunk_registered = {}  # 이번 청크에서 새로 등록된 전역 센트로이드
            for speaker, centroid in speaker_centroids.items():
                if chunk_idx == 0 and len(speaker_registry) == 0:
                    for spk, centroid in speaker_centroids.items():
                        speaker_registry[spk] = centroid
                        mapping[spk] = spk
                        print(f"[INIT][chunk {chunk_idx}] {spk} → {spk}")
                    chunkwise_mapping[chunk_idx] = mapping
                    continue  # 다음 청크로
                # 전역 레지스트리와 이번 청크 신규 전역들 모두와 비교 (dot == cosine)
                best_similarity = -1.0
                best_key = None

                for reg_label, reg_centroid in speaker_registry.items():
                    sim = float(np.dot(reg_centroid, centroid))
                    if sim > best_similarity:
                        best_similarity, best_key = sim, reg_label

                for reg_label, reg_centroid in current_chunk_registered.items():
                    sim = float(np.dot(reg_centroid, centroid))
                    if sim > best_similarity:
                        best_similarity, best_key = sim, reg_label

                if best_similarity >= threshold:
                    mapping[speaker] = best_key
                    print(f"[MAP][chunk {chunk_idx}] {speaker} → {best_key} (sim={best_similarity:.2f})")
                else:
                    new_label = get_next_speaker_name()
                    speaker_registry[new_label] = centroid
                    current_chunk_registered[new_label] = centroid
                    mapping[speaker] = new_label
                    print(f"[NEW][chunk {chunk_idx}] {speaker} → {new_label} (sim={best_similarity:.2f})")
            chunkwise_mapping[chunk_idx] = mapping
        return chunkwise_mapping

    def build_label_mapping_dict_v2(self, chunk_emb_array, threshold=0.6, top_k=3):
        '''
        Build global speaker mapping dict by initializing from the best chunk
        and then matching speakers across all other chunks using embedding similarity.
        
        Parameters:
            - chunk_emb_array: List of tuples like (chunk_idx, emb_array, original_labels, segment_bounds)
            - threshold: cosine similarity threshold for matching
            - top_k: number of initial chunks to consider when choosing best chunk
        
        Returns:
            - chunkwise_mapping: Dict of chunk_idx → {local_label → global_label}
        '''
        # 1. 선택 기준: 가장 많은 화자를 포함한 초기 청크 선택
        def select_initial_chunk(chunks, top_k=3):
            chunk_stats = [
                (chunk_idx, len(set(labels)))
                for chunk_idx, _, labels, _ in chunks[:top_k]
            ]
            best_chunk = max(chunk_stats, key=lambda x: x[1])
            return best_chunk[0]

        def get_next_speaker_name():
            existing_ids = [
                int(re.search(r'\d+', label).group())
                for label in speaker_registry.keys()
                if re.match(r'^SPEAKER_\d+$', label)
            ]
            next_id = max(existing_ids) + 1 if existing_ids else 0
            return f'SPEAKER_{next_id:02d}'

        def compute_centroids(emb_array, labels):
            speaker_to_embs = defaultdict(list)
            for emb, label in zip(emb_array, labels):
                if label == 'UNKNOWN':
                    continue
                speaker_to_embs[label].append(emb)
            return {
                speaker: np.mean(np.stack(embs), axis=0)
                for speaker, embs in speaker_to_embs.items()
            }
        speaker_registry = {}           # global_label: centroid
        chunkwise_mapping = {}          # chunk_idx: {local → global}
        initial_chunk_idx = select_initial_chunk(chunk_emb_array, top_k=top_k)

        # 2. 초기 청크만 먼저 처리하여 global registry 초기화
        for chunk_idx, emb_array, original_labels, _ in chunk_emb_array:
            if chunk_idx != initial_chunk_idx:
                continue
            speaker_centroids = compute_centroids(emb_array, original_labels)
            mapping = {}
            for speaker, centroid in speaker_centroids.items():
                speaker_registry[speaker] = centroid
                mapping[speaker] = speaker
                print(f"[INIT][chunk {chunk_idx}] {speaker} → {speaker}")
            chunkwise_mapping[chunk_idx] = mapping

        for chunk_idx, emb_array, original_labels, _ in chunk_emb_array:
            if chunk_idx == initial_chunk_idx:
                continue

            speaker_centroids = compute_centroids(emb_array, original_labels)
            mapping = {}
            current_chunk_registered = {}
            for speaker, centroid in speaker_centroids.items():
                best_similarity = -1
                best_key = None
                for reg_label, reg_centroid in speaker_registry.items():
                    sim = self.calc_emb_similarity(torch.tensor(reg_centroid), torch.tensor(centroid))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_key = reg_label
                for reg_label, reg_centroid in current_chunk_registered.items():
                    sim = self.calc_emb_similarity(torch.tensor(reg_centroid), torch.tensor(centroid))
                    if sim > best_similarity:
                        best_similarity = sim
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


class EMBPipe(BasePipeline):
    '''
    speaker embedding pipeline
    '''
    def __init__(self, config):
        super().__init__()
        self.wsemb = WSEMB()
        self.emb_model = self.wsemb.load_model(model_path=config['model_path'])
        self.emb_visualizer = EMBVisualizer()

    def get_emb_from_file(self, file_name):
        emb = self.wsemb.get_embedding(self.emb_model, file_name)
        return emb

    def plot_tsne(self, emb_array, labels, save_path=None):
        '''
        emb_array: numpy array of shape (n_samples, n_features)
        labels: list of speaker labels corresponding to each embedding
        '''
        self.emb_visualizer.tsne_and_plot(emb_array, labels, title='test', file_path=save_path)

    def get_xy_tsne(self, emb_array, labels, file_names, as_dict=True):
        return self.emb_visualizer.get_xy_tsne(emb_array, labels=labels, file_names=file_names, as_dict=as_dict)

class VisualizePipe(BasePipeline):
    def __init__(self):
        super().__init__()
        self.audio_visualizer = AudioVisualizer()
    
    def get_melspectrogram(self, audio_file):
        '''
        Get mel spectrogram from audio file
        input:
            - audio_file: audio file path
        output:
            - melspectrogram: numpy array of shape (n_mels, time_steps)
        '''
        return self.audio_visualizer.melspec_png_base64(audio_file)

    def get_waveform(self, audio_file):
        '''
        Get mel spectrogram from audio file
        input:
            - audio_file: audio file path
        output:
            - melspectrogram: numpy array of shape (n_mels, time_steps)
        '''
        return self.audio_visualizer.waveform_png_base64(audio_file)