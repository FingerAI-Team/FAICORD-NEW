from src import SummaryPipe
import requests
import argparse
import json
import os

def main(args):
    url = "https://4c2cf7d77bd3.ngrok-free.app/v1/chat/completion"
    file_name = os.path.join('./dataset/stt/', args.file_name)

    with open('./config/default_system_prompt.txt', "r", encoding="utf-8") as f:
        default_system_prompt = f.read()

    with open('./config/default_subrole_prompt.txt', "r", encoding="utf-8") as f:
        default_subrole_prompt = f.read()

    with open(os.path.join('./config', "concat_system_prompt.txt"), "r", encoding="utf-8") as f:
        concat_system_prompt = f.read()

    summary_pipe = SummaryPipe()
    stt_result = summary_pipe.read_stt_result(file_name)
    stt_chunks = summary_pipe.split_stt_result(stt_result, chunk_count=3)   # [초반부, 중반부, 후반부]

    for chunk in stt_chunks:
        prompt_txt = summary_pipe.build_summary_prompt(concat_system_prompt, chunk)
        data = {
            "question": json.dumps(prompt_txt, ensure_ascii=False)   # JSON 배열을 문자열로 인코딩
        }
        res = requests.post(url, json=data)
        print(res.status_code)
        print(res.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="stt_result.json", help="STT 결과 파일명")
    args = parser.parse_args()
    main(args)