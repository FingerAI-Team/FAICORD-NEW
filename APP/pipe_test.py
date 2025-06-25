from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe
from dotenv import load_dotenv
import numpy as np
import argparse
import json
import time
import os


def main(args):
    '''
    Default Setting
    '''
    start = time.time()
    vad_config = os.path.join(args.model_config_path, 'pyannote_vad_config.yaml')
    diar_config = os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml')
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe(vad_config)
    diar_pipe = DIARPipe(diar_config)
    postprocess_pipe = PostProcessPipe()
    
    '''
    Cleanse audio, Get VAD Result, Get Diar Result, Process Diar Result 
    '''
    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=args.chunk_length, deverve=True)
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    diar_result, _ = diar_pipe.get_diar(args.file_name, return_embeddings=False)   # emb 값 사용 x 
    processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)    # ok. 
    # relabeled_diar = postprocess_pipe.relabel_nonoverlapped_labels(args.file_name, non_overlapped_diar)
    chunk_emb_array = postprocess_pipe.get_chunk_emb_array(args.file_name, non_overlapped_diar)
    label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
    print(label_mapping_dict)
    full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
    final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)
    diar_pipe.save_files(final_diar, file_name=args.file_name)
    file_name = args.file_name.split('/')[-1].split('.')[0] 
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)