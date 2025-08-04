from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, STTPipe, SummaryPipe
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
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
    frontend_pipe = FrontendPipe()
    vad_pipe = VADPipe(vad_config)
    diar_pipe = DIARPipe(diar_config)
    postprocess_pipe = PostProcessPipe()
   
    clean_audio = frontend_pipe.process_audio(args.file_name, chunk_length=args.chunk_length, deverve=True)
    print(f'cleanse time: {time.time() - start}')
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    diar_result, _ = diar_pipe.get_diar(args.file_name, return_embeddings=False)   # emb 값 사용 x 
    chunk_emb_array = postprocess_pipe.get_chunk_emb_array(args.file_name, diar_result)
    # print(chunk_emb_array)
    # cosine distance 계산
    dist_mat = cosine_distances(chunk_emb_array[0][1])
    agglo = AgglomerativeClustering(
        metric='precomputed',            # ✅ affinity → metric
        linkage='average',
        distance_threshold=0.5,
        n_clusters=None
    )
    cluster_labels = agglo.fit_predict(dist_mat)
    num_clusters = len(set([label for label in cluster_labels if label != -1]))
    print(f"Detected {num_clusters} speakers")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--whisper_config_path', type=str, default='./config')
    cli_parser.add_argument('--model_config_path', type=str, default='./models')
    cli_parser.add_argument('--file_name', type=str, required=True)
    cli_parser.add_argument('--chunk_length', type=int, default=300)
    cli_args = cli_parser.parse_args()
    main(cli_args)