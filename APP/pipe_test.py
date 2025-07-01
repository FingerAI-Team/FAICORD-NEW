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
    
    whisper_api = os.getenv('OPENAI_API')
    audio_file_name = os.path.join('./dataset/audio/', args.file_name.split('.')[0] + '.wav')
    stt_pipe = STTPipe()
    stt_pipe.set_env(whisper_api=whisper_api, generation_config=generation_config)
    # print(audio_file_name)
    # whisper_audio = stt_pipe.prepare_audio(audio_file_name)
    diar_result = stt_pipe.read_rttm(os.path.join('./dataset/rttm/', args.file_name))
    stt_result = stt_pipe.transcribe_by_rttm(audio_file_name, diar_result)
    print(stt_result)

    # stt_result = stt_pipe.transcribe_text(args.file_name, vad_result=vad_merged, transcribe_type='api')
    save_file_name = 'faicord_' + args.file_name.split('/')[-1].split('.')[0] + '.json'
    with open(os.path.join('./dataset/stt/', save_file_name), "w", encoding="utf-8") as f:
        json.dump(stt_result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)