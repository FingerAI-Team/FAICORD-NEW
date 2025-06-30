from src import FrontendPipe, VADPipe, PostProcessPipe, STTPipe
from dotenv import load_dotenv
import numpy as np
import argparse 
import json 
import os 

def main(args):
    load_dotenv()
    with open(os.path.join(args.whisper_config_path, 'generation_config.json')) as f: 
        generation_config = json.load(f)
    
    whisper_api = os.getenv('WHISPER_API')
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe()
    stt_pipe = STTPipe()
    
    frontend_pipe.set_env()
    vad_pipe.set_env(os.path.join(args.model_config_path, 'pyannote_vad_config.yaml'))
    stt_pipe.set_env(whisper_api=whisper_api, generation_config=generation_config)
    
    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=300, deverve=True)
    # frontend_pipe.save_audio(clean_audio, 'frontend-processed.wav')
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    # print(vad_result)
    vad_merged = vad_pipe.merge_segments(vad_result, min_length=5, silence_gap=5, min_keep_length=0.5)
    print(vad_merged)
    stt_result = stt_pipe.transcribe_text(args.file_name, vad_result=vad_merged, transcribe_type='api')
    save_file_name = 'stt_origin_vad_' + args.file_name.split('/')[-1].split('.')[0] + '.txt'
    stt_output = stt_pipe.postprocess_result(stt_result, file_name=os.path.join('./dataset/stt/', save_file_name))
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)