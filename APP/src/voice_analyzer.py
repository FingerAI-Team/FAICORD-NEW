'''
Synapsoft STT 서비스 활용 모듈 
'''
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
        response = requests.post('http://127.0.0.1:8000/asr', files=files)
        return response.json()
