from src import FileProcessor, DataProcessor, VoiceAnalyzer
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
    
    api_key = os.getenv("STT_API_KEY")
    local_path = './dataset/audio'
    va_path = './data/samples/audio'
    local_file_path = os.path.join(local_path, args.file_name)
    
    analyzer = VoiceAnalyzer(api_key=api_key)
    ''' load test & answer file '''
    file_p = FileProcessor()
    whisper_test_data = file_p.load_json(local_file_path)
    synap_test_data = file_p.load_json(local_file_path.replace('.json', '_va.json'))
    answer_data = file_p.load_json(local_file_path.replace('.json', '_정답.json'))

    print("whisper stt type:", type(whisper_test_data))
    print("synap stt type", type(synap_test_data))
    print("answer type:", type(answer_data))

    ''' calculate CER '''
    data_p = DataProcessor()
    print(f'synap test data length: {len(synap_test_data)}')
    synap_new_data = data_p.map_timeline(synap_test_data, answer_data)
    print(f'synap_new: {len(synap_new_data)}')
    analyzer.save_result_to_json(processd_segments=synap_new_data, save_path=local_file_path.replace('.json', '_va_mapped.json'))
    cer_whisper = data_p.calc_cer(whisper_test_data, answer_data)
    cer_synap = data_p.calc_cer(synap_new_data, answer_data)
    print(f"Whisper CER: {cer_whisper:.4f}")
    print(f"Synap CER: {cer_synap:.4f}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Voice Analyzer Test")
    cli_parser.add_argument("--file_name", type=str, required=True, help="Path to the audio file to be analyzed")
    cli_parser.add_argument("--input_type", type=str, default="local", help="Type of input (default: local)")
    main_args = cli_parser.parse_args()
    main(main_args)