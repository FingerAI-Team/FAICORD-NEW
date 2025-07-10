from markdown_pdf import Section
from dotenv import load_dotenv
from src import SummaryPipe
import markdown
import argparse
import json
import os
 

def convert_minutes_to_markdown(raw_text: str) -> str:
    lines = raw_text.strip().split('\n')
    md_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            md_lines.append('')
        elif line.startswith('## '):  # 예: ## 안건
            md_lines.append(f'# {line[3:].strip()}')
        elif line.startswith('### '):  # 예: ### 안건 1:
            md_lines.append(f'## {line[4:].strip()}')
        elif line.startswith('- '):  # Bullet point
            md_lines.append(f'- {line[2:].strip()}')
        elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.'):
            md_lines.append(f'{line.strip()}')  # 번호 붙은 항목은 그대로
        elif line.startswith('**') and '**' in line[2:]:  # 발언자 라인
            md_lines.append(f'- {line}')
        else:
            md_lines.append(line)
    return '\n'.join(md_lines)


def main(args):
    '''
    Default Setting
    '''
    load_dotenv()
    with open(os.path.join(args.whisper_config_path, 'generation_config.json')) as f: 
        generation_config = json.load(f)
    
    print(f'Generation Config: {generation_config}')
    summary_pipe = SummaryPipe(config=generation_config, api_key=os.getenv('OPENAI_API'))
    openai_summary_model = summary_pipe.set_openai_client()
    with open(args.file_name, 'r', encoding='utf-8') as f:
        stt_result = json.load(f)  
    summary_result = summary_pipe.summarize(openai_summary_model, stt_result) 
    # print(summary_result) 
    markdown_text = convert_minutes_to_markdown(summary_result)
    save_file_name = 'faicord_' + args.file_name.split('/')[-1].split('.')[0] + '_summary.html'
    with open(save_file_name, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
        
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
        f.write(html_text)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)