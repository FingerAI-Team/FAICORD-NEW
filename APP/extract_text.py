import json 
import os 
from dotenv import load_dotenv
from src.pipe import STTPipe
import argparse

def main(args):
    load_dotenv()
    with open(os.path.join('./dataset/stt/', args.file_name), 'r', encoding='utf-8') as f:
        stt_result = json.load(f)

    text_list = '' 
    for result in stt_result:
        text_list += result['text'] + ' '
    
    with open(os.path.join('./dataset/stt/', args.file_name.replace('.json', '.txt')), 'w', encoding='utf-8') as f:
        f.write(text_list.strip())

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)    