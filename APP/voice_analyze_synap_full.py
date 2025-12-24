from src import VoiceAnalyzer
from dotenv import load_dotenv
import argparse
import os

def main(args):
    load_dotenv()
    api_key = os.getenv("STT_API_KEY")
    analyzer = VoiceAnalyzer(api_key=api_key)
    result = analyzer.analyze(file_path=args.file_path, input_type=args.input_type)
    save_file_name = args.file_path.split('/')[-1].split('.')[0] + '_va_full.json'   # va: voice analyzer
    analyzer.save_result_to_json(analysis_result=result, save_path=os.path.join('./dataset/stt/',save_file_name))

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Voice Analyzer Test")
    cli_parser.add_argument("--file_path", type=str, required=True, help="Path to the audio file to be analyzed")
    cli_parser.add_argument("--input_type", type=str, default="local", help="Type of input (default: local)")
    main_args = cli_parser.parse_args()
    main(main_args)