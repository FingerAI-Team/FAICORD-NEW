from .stt import WhisperSTT
from .llms import LLMOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cosine
from collections import defaultdict
from abc import abstractmethod
from pydub import AudioSegment
from datetime import datetime
from pathlib import Path
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
    def __init__(self, config=None, api_key=None):
        super().__init__()
        if api_key != None:
            self.config = config   
            self.api_key = api_key

    def set_openai_client(self):
        openai_summary_model = LLMOpenAI(config=self.config, api_key=self.api_key)
        return openai_summary_model
    
    def build_summary_prompt(self, base_prompt, stt_dialogues: list[dict]) -> str:
        dialogue_str = "\n".join(
            f'{item["speaker"]}: {item["text"].strip()}' for item in stt_dialogues if item.get("text")
        )
        return f"{base_prompt.strip()}\n\n---\n\n아래는 회의 대화록 전체입니다:\n\n{dialogue_str}"

    def convert_to_train_format(self, file_path=None, stt_result=None, target_summary=None, output_path=None):
        speakers = sorted(list({u["speaker"] for u in stt_result if "speaker" in u}))
        transcript = [{"speaker": u["speaker"], "text": u["text"]} for u in stt_result if "text" in u]

        file_id = Path(file_path).stem
        date_str = datetime.now().strftime("%Y-%m-%d")
        full_id = f"{file_id}_{date_str}"
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
                continue    # 빈 문자열은 제외
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

    def convert_to_multi_segment_train_samples(self, stt_result, summary_seg_result, target_summary, file_id=None, output_path=None, segment_count=3):
        speakers = sorted(list({u["speaker"] for u in stt_result if "speaker" in u}))
        transcript = [{"speaker": u["speaker"], "text": u["text"]} for u in stt_result if "text" in u]
        segments = self.split_stt_result(stt_result, chunk_count=segment_count)
       
        date_str = datetime.now().strftime("%Y-%m-%d")
        samples = []

        # 1차 요약용 샘플 생성
        for i, seg in enumerate(segments):
            seg_lines = "\n".join([f'{u["speaker"]}: {u["text"]}' for u in seg])
            input_text = f"<|user|>\n아래 전사 블록을 간단히 요약하세요:\n\n{seg_lines}"
            samples.append({
                "id": f"{file_id}_seg{i+1}_{date_str}",
                "input": input_text,
                "target_summary": summary_seg_result[i]
            })

        # 2차 통합 요약용 샘플 생성
        seg_refs = [f"요약 {i+1}: {summary_seg_result[i]}" for i in range(segment_count)]
        doc_input = "<|user|>\n아래 요약들을 종합하여 회의록을 정리하세요:\n\n" + "\n".join(seg_refs)
        samples.append({
            "id": f"{file_id}_doc_{date_str}",
            "input": doc_input,
            "target_summary": target_summary 
        })
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[✓] {len(samples)}개의 학습 샘플이 저장되었습니다 → {output_path}")
    
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