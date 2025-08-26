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
    
    def convert_to_train_format(input_path=None, target_summary=None, output_path=None):
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        speakers = sorted(list({u["speaker"] for u in raw_data if "speaker" in u}))
        transcript = [{"speaker": u["speaker"], "text": u["text"].strip()} for u in raw_data if "text" in u]

        file_id = Path(input_path).stem
        date_str = datetime.now().strftime("%Y-%m-%d")
        full_id = f"{file_id}_{date_str}"
        '''target_summary = (
            "안건\n"
            "음성 텍스트 변환 프로젝트 진행 상황\n"
            "유사도 분석 및 스크립트화 작업\n"
            "데이터 제공 및 테스트 계획\n"
            "논의 사항\n"
            "안건 1: ..."
        )'''
        target_summary = target_summary if target_summary else "추가 예정"
        output = {
            "id": full_id,
            "speakers": speakers,
            "transcript": transcript,
            "meta": {
                "domain": "STT/문서 자동화",
                "language": "ko"
            },
            "target_summary": target_summary
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def read_stt_result(self, stt_path):
        with open(stt_path, "r", encoding="utf-8") as f:
            stt_result = json.load(f)
        # start, end 제거하고 speaker + text만 추출
        dialogue_list = []
        for item in stt_result:
            speaker = item.get("speaker")
            text = item.get("text", "")
            if not text:
                continue  # 빈 문자열은 제외
            dialogue_list.append({
                "speaker": speaker,
                "text": text
            })
        return dialogue_list

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