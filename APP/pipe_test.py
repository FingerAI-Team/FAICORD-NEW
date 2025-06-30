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
    stt_pipe = STTPipe()
    stt_pipe.set_env(whisper_api=whisper_api, generation_config=generation_config)
    diar_result = 
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