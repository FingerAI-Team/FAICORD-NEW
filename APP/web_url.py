from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, STTPipe, SummaryPipe
from flask import Flask, send_file, request, jsonify, Response
from dotenv import load_dotenv
import tempfile
import requests
import markdown
import logging
import time
import json
import os

app = Flask(__name__)

load_dotenv()
vad_config = os.path.join('./models', 'pyannote_vad_config.yaml')
diar_config = os.path.join('./models', 'pyannote_diarization_config.yaml')
frontend_pipe = FrontendPipe()
vad_pipe = VADPipe(vad_config)
diar_pipe = DIARPipe(diar_config)
postprocess_pipe = PostProcessPipe()
with open('./config/generation_config.json') as f:
    generation_config = json.load(f)
whisper_api = os.getenv('OPENAI_API')
stt_pipe = STTPipe(whisper_api=whisper_api, generation_config=generation_config)
summary_pipe = SummaryPipe(config=generation_config, api_key=os.getenv('OPENAI_API'))
openai_summary_model = summary_pipe.set_openai_client()

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    audio_file = request.files['file']
    chunk_length = int(request.form.get('chunk_length', 300))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    try:
        start = time.time()

        # 1. Preprocess
        clean_audio = frontend_pipe.process_audio(tmp_path, chunk_length=chunk_length, deverve=True)
        vad_result = vad_pipe.get_vad_timestamp(clean_audio)
        diar_result, _ = diar_pipe.get_diar(tmp_path, return_embeddings=False)
        processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)

        # 2. Label mapping
        chunk_emb_array = postprocess_pipe.get_chunk_emb_array(tmp_path, non_overlapped_diar)
        label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
        full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
        final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)

        # 3. Save RTTM and run STT
        diar_pipe.save_merged_rttm(final_diar, tmp_path)
        rttm_file = tmp_path.replace('/audio/', '/rttm/').replace('.wav', '.rttm')
        diar_result = stt_pipe.read_rttm(rttm_file)
        stt_result = stt_pipe.transcribe_by_rttm(tmp_path, diar_result)
        summary_result = summary_pipe.summarize(openai_summary_model, stt_result) 
        print(f'Summarize Done !: {time.time() - start}초')
        
        markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
        save_file_name = 'faicord_' + audio_file.split('/')[-1].split('.')[0] + '_summary.html'    
        html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
        with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
            f.write(html_text)
        return jsonify({'status': 'success', 'data': stt_result, 'time': round(time.time() - start, 2)})   
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(tmp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
