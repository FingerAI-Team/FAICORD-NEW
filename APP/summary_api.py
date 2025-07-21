from markdown_pdf import Section
from dotenv import load_dotenv
from src import SummaryPipe
import argparse
import markdown
import json
import os


def main(args):
    '''
    Default Setting
    '''
    load_dotenv()
    with open(os.path.join(args.whisper_config_path, 'generation_config.json')) as f: 
        generation_config = json.load(f)
    
    with open(os.path.join('./config', "default_system_prompt.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    with open(os.path.join('./config', "default_subrole_prompt.txt"), "r", encoding="utf-8") as f:
        subrole_prompt = f.read()

    print(f'Generation Config: {generation_config}')
    summary_pipe = SummaryPipe(config=generation_config, api_key=os.getenv('OPENAI_API'))
    openai_summary_model = summary_pipe.set_openai_client()
    with open(args.file_name, 'r', encoding='utf-8') as f:
        stt_result = json.load(f)  
    # print(stt_result)
    summary_result = summary_pipe.summarize(openai_summary_model, stt_result, system_prompt=system_prompt, subrole_prompt=subrole_prompt) 
    markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
    save_file_name = 'faicord_' + args.file_name.split('/')[-1].split('.')[0] + '_summary.html'    
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
        f.write(html_text)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)