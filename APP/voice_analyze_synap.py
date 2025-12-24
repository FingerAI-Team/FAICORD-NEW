from src import VoiceAnalyzer, STTPipe
from dotenv import load_dotenv
import argparse
import os

def main(args):
    load_dotenv()
    api_key = os.getenv("STT_API_KEY")
    analyzer = VoiceAnalyzer(api_key=api_key)
    stt_pipe = STTPipe()
    ''' load rttm file '''
    rttm_file = args.file_name.replace('/audio/', '/rttm/').replace('.wav', '.rttm')
    rttm_result = stt_pipe.read_rttm(rttm_file)
    ''' analyze with diar info '''
    processed_rttm = analyzer.process_rttm_for_stt(rttm_result)
    result = analyzer.analyze(file_path=args.file_path, input_type=args.input_type, timeline=processed_rttm)
    save_file_name = args.file_path.split('/')[-1].split('.')[0] + '_va.json'   # va: voice analyzer
    analyzer.save_result_to_json(analysis_result=result, save_path=os.path.join('./dataset/stt/',save_file_name))

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Voice Analyzer Test")
    cli_parser.add_argument("--file_path", type=str, required=True, help="Path to the audio file to be analyzed")
    cli_parser.add_argument("--input_type", type=str, default="local", help="Type of input (default: local)")
    main_args = cli_parser.parse_args()
    main(main_args)