from src import VoiceAnalyzer
from dotenv import load_dotenv
import argparse
import json
import os

def main(args):
    load_dotenv()
    api_key = os.getenv("STT_API_KEY")
    analyzer = VoiceAnalyzer(api_key=api_key)
    result = analyzer.analyze(file_path=args.file_path, input_type=args.input_type)
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Voice Analyzer Test")
    cli_parser.add_argument("--file_path", type=str, required=True, help="Path to the audio file to be analyzed")
    cli_parser.add_argument("--input_type", type=str, default="local", help="Type of input (default: local)")
    main_args = cli_parser.parse_args()
    main(main_args)