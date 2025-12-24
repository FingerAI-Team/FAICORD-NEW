from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, STTPipe, SummaryPipe, EMBPipe
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import argparse
import markdown
import time
import json
import os
import numpy as np

def main(args):
    '''
    Default Setting
    '''
    start = time.time()
    load_dotenv()
    with open(os.path.join(args.whisper_config_path, 'generation_config.json')) as f: 
        generation_config = json.load(f)

    with open(os.path.join(args.model_config_path, 'wespeak_config.json')) as f: 
        emb_config = json.load(f)

    vad_config = os.path.join(args.model_config_path, 'pyannote_vad_config.yaml')
    diar_config = os.path.join(args.model_config_path, 'pyannote_diarization_config.yaml')
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe(vad_config)
    diar_pipe = DIARPipe(diar_config)
    postprocess_pipe = PostProcessPipe()
    emb_pipe = EMBPipe(emb_config)

    # print(np.shape(emb_pipe.get_emb_from_file(args.file_name)))
    # speaker_emb = emb_pipe.get_emb_from_file(args.file_name)
    speaker_emb2 = emb_pipe.get_emb_from_file('./dataset/audio/원라인.wav')
    speaker_emb3 = emb_pipe.get_emb_from_file('./dataset/audio/김태완매니저3.wav')
    # embeddings = np.vstack([speaker_emb, speaker_emb2, speaker_emb3])
    # labels = ['speaker_a', 'speaker_b', 'speaker_a']
    # emb_pipe.plot_tsne(embeddings, labels, save_path='./dataset/emb/')
    # print(speaker_emb)
    
    sim = cosine_similarity(
        speaker_emb3.reshape(1, -1),
        speaker_emb2.reshape(1, -1)
    )[0][0]
    print(f"Cosine similarity: {sim:.4f}")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--emb_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)