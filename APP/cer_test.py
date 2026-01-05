from src import FileProcessor, DataProcessor
from dotenv import load_dotenv
import argparse
import markdown
import time 
import json
import os

def main(args):
    load_dotenv()
    local_path = './dataset/testset'
    local_file_path = os.path.join(local_path, args.file_name)
    
    ''' load test & answer file '''
    file_p = FileProcessor()
    whisper_test_data = file_p.load_json(local_file_path)
    synap_test_data = file_p.load_json(local_file_path.replace('.json', '_va.json'))
    answer_data = file_p.load_json(local_file_path.replace('.json', '_정답.json'))

    ''' calculate CER '''
    data_p = DataProcessor()
    print(f'whisper_stt: {whisper_test_data}')
    whisper_stt = whisper_test_data['result']
    synap_stt = synap_test_data['result']
    cer_whisper = data_p.calc_cer(whisper_stt, answer_data)
    cer_synap = data_p.calc_cer(synap_stt, answer_data)
    print(f"Whisper CER: {cer_whisper:.4f}")
    print(f"Synap CER: {cer_synap:.4f}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Voice Analyzer Test")
    cli_parser.add_argument("--file_name", type=str, required=True, help="Path to the audio file to be analyzed")
    cli_parser.add_argument("--input_type", type=str, default="local", help="Type of input (default: local)")
    main_args = cli_parser.parse_args()
    main(main_args)