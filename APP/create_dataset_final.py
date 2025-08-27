from dotenv import load_dotenv
from src import SummaryPipe
import argparse
import json
import os 

def main(args):
    load_dotenv()
    openai_api = os.getenv("OPENAI_API")
    with open('./config/generation_config.json') as f:
        generation_config = json.load(f)

    with open('./config/default_system_prompt.txt', "r", encoding="utf-8") as f:
        default_system_prompt = f.read()

    with open('./config/default_subrole_prompt.txt', "r", encoding="utf-8") as f:
        default_subrole_prompt = f.read()

    with open(os.path.join('./config', "concat_system_prompt.txt"), "r", encoding="utf-8") as f:
        concat_system_prompt = f.read()

    summary_pipe = SummaryPipe(config=generation_config, api_key=openai_api)
    summary_model = summary_pipe.set_openai_client()
    stt_result = summary_pipe.read_stt_result(args.stt_file_name)
    stt_results = summary_pipe.split_stt_result(stt_result, chunk_count=3)  # [초반부, 중반부, 후반부]
    
    chunk_summary = '' 
    summary_seg_result = []
    for idx in range(len(stt_results)):
        summary_result = summary_pipe.summarize(summary_model, stt_results[idx], system_prompt=default_system_prompt, subrole_prompt=default_subrole_prompt) 
        summary_seg_result.append(summary_result)
        chunk_summary += summary_result + '\n\n'
    total_summary = summary_pipe.summarize(summary_model, chunk_summary, system_prompt=concat_system_prompt, subrole_prompt='')   
    output_file_name = args.stt_file_name.replace('.json', '_summary.json').replace('/stt/', '/summary/')
    summary_pipe.convert_to_multi_segment_train_samples(stt_result=stt_result, summary_seg_result=summary_seg_result, target_summary=total_summary, file_id='test', output_path=output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt_file_name", type=str, default="stt_result.json", help="STT 결과 파일명")
    args = parser.parse_args()
    main(args)