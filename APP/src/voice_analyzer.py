'''
Synapsoft STT 서비스 활용 모듈 
'''
from email.mime import text
import json
import requests

class VoiceAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def analyze(self, file_path: str, input_type: str = 'local'):
        files = {
            'api_key': (None, self.api_key),
            'input_type': (None, input_type),
            'file_path': (None, file_path),
        }
        response = requests.post('http://voice-analyzer-v2:8000/asr', files=files)
        return response.json()

    def analyze_with_timeline(self, file_path: str, input_type: str = 'local', timeline: list = None):
        files = {
            'api_key': (None, self.api_key),
            'input_type': (None, input_type),
            'file_path': (None, file_path),
            'no_forcedalign': (None, 'true'),
            'no_diarization': (None, 'true'),
            'segment_timeline': (None, json.dumps(timeline)),
        }
        response = requests.post('http://voice-analyzer-v2:8000/asr', files=files)
        return response.json()

    def extract_text(self, analysis_result: dict):
        return analysis_result["result"]["text"]  
        
    def process_rttm_for_stt(self, rttm_df):
        rttm_df["end"] = rttm_df["start"] + rttm_df["duration"]
        print(f"timeline: {rttm_df[['start', 'end']].values.tolist()}")
        return rttm_df[["start", "end"]].values.tolist()

    def save_result_to_json(self, analysis_result: dict, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)