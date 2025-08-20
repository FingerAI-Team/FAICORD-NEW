from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, STTPipe, SummaryPipe
from dotenv import load_dotenv
import argparse
import markdown
import time
import json
import os


def main(args):
    '''
    Default Setting
    '''
    start = time.time()
    load_dotenv()
    with open(os.path.join(args.whisper_config_path, 'generation_config.json')) as f: 
        generation_config = json.load(f)

    vad_config = os.path.join(args.model_config_path, 'pyannote_vad_config.yaml')
    diar_config = os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml')
    with open(os.path.join('./config', "default_system_prompt.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    with open(os.path.join('./config', "default_subrole_prompt.txt"), "r", encoding="utf-8") as f:
        subrole_prompt = f.read()
    
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe(vad_config)
    diar_pipe = DIARPipe(diar_config)
    postprocess_pipe = PostProcessPipe()
    rttm_file = args.file_name.replace('/audio/', '/rttm/').replace('.wav', '.rttm')
    whisper_api = os.getenv('OPENAI_API')
    stt_pipe = STTPipe(whisper_api=whisper_api, generation_config=generation_config)
    summary_pipe = SummaryPipe(config=generation_config, api_key=os.getenv('OPENAI_API'))
    openai_summary_model = summary_pipe.set_openai_client()
    '''
    Cleanse audio, Get VAD Result, Get Diar Result, Process Diar Result 
    '''
    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=args.chunk_length, deverve=True)
    print(f'cleanse time: {time.time() - start}')
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    diar_result, _ = diar_pipe.get_diar(args.file_name, return_embeddings=False)   # emb 값 사용 x 
    processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)    # ok. 
    # relabeled_diar = postprocess_pipe.relabel_nonoverlapped_labels(args.file_name, non_overlapped_diar)
    
    chunk_emb_array = postprocess_pipe.get_chunk_emb_array(args.file_name, non_overlapped_diar)
    label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
    # print(label_mapping_dict)
    
    full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
    final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)
    rttm_file = args.file_name.replace('/audio/', '/rttm/').replace('.wav', '.rttm')
    diar_pipe.save_merged_rttm(final_diar, file_name=rttm_file)
    print(f'Diarization Done !: {time.time() - start}초')
    
    diar_result = stt_pipe.read_rttm(rttm_file)
    stt_result = stt_pipe.transcribe_by_rttm(args.file_name, diar_result)
    print(f'STT Done !: {time.time() - start}초')
    save_file_name = 'faicord_' + args.file_name.split('/')[-1].split('.')[0] + '.json'
    with open(os.path.join('./dataset/stt/', save_file_name), "w", encoding="utf-8") as f:
        json.dump(stt_result, f, ensure_ascii=False, indent=2)
    
    summary_result = summary_pipe.summarize(openai_summary_model, stt_result, system_prompt=system_prompt, subrole_prompt=subrole_prompt) 
    print(f'Summarize Done !: {time.time() - start}초')
    markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
    save_file_name = 'faicord_' + args.file_name.split('/')[-1].split('.')[0] + '_summary.html'    
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
        f.write(html_text)
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)