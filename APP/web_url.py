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
file_path = os.getenv('FILE_PATH', './dataset/audio/')
vad_config = os.path.join('./models', 'pyannote_vad_config.yaml')
diar_config = os.path.join('./models', 'pyannote_diarization_config.yaml')

frontend_pipe = FrontendPipe()
vad_pipe = VADPipe(vad_config)
diar_pipe = DIARPipe(diar_config)
postprocess_pipe = PostProcessPipe()
with open('./config/generation_config.json') as f:
    generation_config = json.load(f)

with open(os.path.join('./config', "default_system_prompt.txt"), "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open(os.path.join('./config', "default_subrole_prompt.txt"), "r", encoding="utf-8") as f:
    subrole_prompt = f.read()

whisper_api = os.getenv('OPENAI_API')
stt_pipe = STTPipe(whisper_api=whisper_api, generation_config=generation_config)
summary_pipe = SummaryPipe(config=generation_config, api_key=os.getenv('OPENAI_API'))
openai_summary_model = summary_pipe.set_openai_client()

@app.route('/summarize_audio', methods=['POST'])
def summarize_audio():
    if 'file_name' not in request.files:
        return jsonify({'status': "upload failed", 'message': f'Invalid file path: {file_path}'}), 400
    start = time.time()
    audio_file = request.files['file_name']
    user_system_prompt = request.form.get('system_prompt', 'default')
    user_subrole_prompt = request.form.get('subrole_prompt', 'default')

    meeting_log = request.files['meeting_log']
    if user_system_prompt == 'default' and user_subrole_prompt == 'default':
        summary_result = summary_pipe.summarize(openai_summary_model, meeting_log, system_prompt=system_prompt, subrole_prompt=subrole_prompt) 
    elif user_system_prompt == 'default' and user_subrole_prompt != 'default':
        summary_result = summary_pipe.summarize(openai_summary_model, meeting_log, system_prompt=system_prompt, subrole_prompt=user_subrole_prompt)
    elif user_system_prompt != 'default' and user_subrole_prompt == 'default':
        summary_result = summary_pipe.summarize(openai_summary_model, meeting_log, system_prompt=user_system_prompt, subrole_prompt=subrole_prompt)
    elif user_system_prompt != 'default' and user_subrole_prompt != 'default':
        summary_result = summary_pipe.summarize(openai_summary_model, meeting_log, system_prompt=user_system_prompt, subrole_prompt=subrole_prompt)
    print(f'Summarize Done !: {time.time() - start}초')
    
    markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
    save_file_name = 'faicord_' + audio_file.split('/')[-1].split('.')[0] + '_summary.html'    
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
        f.write(html_text)
    return jsonify({'status': 'success', 'data': meeting_log})   


@app.route('/process_audio', methods=['POST'])
def process_audio():
    if request.is_json:
        data = request.get_json()
        file_path = data.get('file_name')
    else:
        file_path = request.form.get('file_name')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'status': 'file upload failed', 'message': f'Invalid file path: {file_path}'}), 400
    start = time.time()
    clean_audio = frontend_pipe.process_audio(file_path, chunk_length=300, deverve=True)
    vad_result = vad_pipe.get_vad_timestamp(clean_audio)
    try: 
        diar_result, _ = diar_pipe.get_diar(file_path, return_embeddings=False)
        processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)
        chunk_emb_array = postprocess_pipe.get_chunk_emb_array(file_path, non_overlapped_diar)
        label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
        full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
        final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)
        print(f'Diarization Done !: {time.time() - start}초')
    except Exception as e:
        logging.error(f"Diarization failed: {e}")
        return jsonify({'status': 'Diarization failed', 'message': str(e), 'status_code': 101}), 500
    rttm_file = file_path.replace('.wav', '.rttm')
    diar_pipe.save_merged_rttm(final_diar, os.path.join('./dataset/rttm/', file_path))
    try:
        diar_result = stt_pipe.read_rttm(rttm_file)
        stt_result = stt_pipe.transcribe_by_rttm(file_path, diar_result)
        print(f'STT Done !: {time.time() - start}초')
    except Exception as e:
        logging.error(f"STT failed: {e}")
        return jsonify({'status': 'STT failed', 'message': str(e), 'status_code': 102}), 500
    
    stt_file = file_path.replace('.wav', '.json')
    with open(os.path.join('./dataset/stt/', stt_file), "w", encoding="utf-8") as f:
        json.dump(stt_result, f, ensure_ascii=False, indent=2)
    try:
        summary_result = summary_pipe.summarize(openai_summary_model, stt_result) 
        print(f'Summarize Done !: {time.time() - start}초')
        markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
        save_file_name = 'faicord_' + file_path.split('.')[0] + '_summary.html'    
        html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
        with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
            f.write(html_text)
    except Exception as e:
        logging.error(f"Summary failed: {e}")
        return jsonify({'status': 'Summary failed', 'message': str(e), 'status_code': 103}), 500
    return jsonify({'status': 'success', 'status_code': 100}), 200   


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)