from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, EMBPipe
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv
import numpy as np
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
    with open(os.path.join(args.model_config_path, 'wespeak_config.json')) as f: 
        emb_config = json.load(f)

    vad_config = os.path.join(args.model_config_path, 'pyannote_vad_config.yaml')
    diar_config = os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml')
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe(vad_config)
    diar_pipe = DIARPipe(diar_config)
    postprocess_pipe = PostProcessPipe()
    emb_pipe = EMBPipe(emb_config)

    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=args.chunk_length, deverve=True)
    print(f'cleanse time: {time.time() - start}')
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    if args.file_name.endswith('.m4a'):
        wav_file_name = args.file_name.replace('.m4a', '.wav')
        diar_result, _ = diar_pipe.get_diar(wav_file_name, return_embeddings=False)   # emb 값 사용 x 
    else:
        diar_result, _ = diar_pipe.get_diar(args.file_name, return_embeddings=False)   # emb 값 사용 x
    diar_pipe.save_merged_rttm(diar_result, file_name='test.rttm')
    # print(diar_result)
    processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)    # ok. 
    diar_pipe.save_merged_rttm(non_overlapped_diar, file_name='no_test.rttm')
    # print(processed_diar)
    # relabeled_diar = postprocess_pipe.relabel_nonoverlapped_labels(wav_file_name, non_overlapped_diar)

    chunk_emb_array = postprocess_pipe.get_chunk_emb_array(wav_file_name, non_overlapped_diar)
    label_mapping_dict, speaker_registry = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
    # print(label_mapping_dict)
    full_diar = postprocess_pipe.apply_labels_to_full_diar_with_embedding(processed_diar, non_overlapped_diar, label_mapping_dict, wav_file_name, speaker_registry)
    # full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
    final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)
    chunk_emb_array2 = postprocess_pipe.get_chunk_emb_array(wav_file_name, final_diar)
    
    all_embeddings = []
    all_labels = []
    for _, embeddings, labels, _ in chunk_emb_array2:
        all_embeddings.extend(embeddings)
        all_labels.extend(labels)
    emb_pipe.plot_tsne(all_embeddings, all_labels, save_path='./dataset/emb/')


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--emb_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)