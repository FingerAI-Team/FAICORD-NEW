from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, STTPipe, SummaryPipe, EMBPipe, VisualizePipe
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np 
import requests
import markdown
import logging 
import uvicorn
import base64
import time
import json
import os
import io 

load_dotenv()

LOG_DIR = 'logs'
LOG_FILE = "app.log" 
os.makedirs(LOG_DIR, exist_ok=True)
log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
logger.propagate = False

app = FastAPI()
file_path = os.getenv('FILE_PATH', './dataset/audio/')
vad_config = os.path.join('./models', 'pyannote_vad_config.yaml')
diar_config = os.path.join('./models', 'pyannote_diarization_config.yaml')

with open('./config/generation_config.json') as f:
    generation_config = json.load(f)

with open('./config/default_system_prompt.txt', "r", encoding="utf-8") as f:
    default_system_prompt = f.read()

with open('./config/default_subrole_prompt.txt', "r", encoding="utf-8") as f:
    default_subrole_prompt = f.read()

with open(os.path.join('./config', "concat_system_prompt.txt"), "r", encoding="utf-8") as f:
    concat_system_prompt = f.read()

with open(os.path.join("./models", 'wespeak_config.json')) as f: 
    emb_config = json.load(f)

class VisualizeEmbRequest(BaseModel):
    audio_file_list: List[str]
    label_list: List[str]

class MelReq(BaseModel):
    audio_file: str

class WaveformReq(BaseModel):
    audio_file: str

frontend_pipe = FrontendPipe()
vad_pipe = VADPipe(vad_config)
diar_pipe = DIARPipe(diar_config)
emb_pipe = EMBPipe(emb_config)
postprocess_pipe = PostProcessPipe()
visualize_pipe = VisualizePipe()

whisper_api = os.getenv('OPENAI_API')
stt_pipe = STTPipe(whisper_api=whisper_api, generation_config=generation_config)
summary_pipe = SummaryPipe(config=generation_config, api_key=whisper_api)
openai_summary_model = summary_pipe.set_openai_client()

@app.post("/process_audio")
async def process_audio_endpoint(
    background_tasks: BackgroundTasks,
    meeting_dir: Optional[str] = Form(None),
    file_name: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None)
):
    print(f'file_name: {file_name}')
    logger.info(f"[{file_name}]")
    try:
        # Background task 등록
        background_tasks.add_task(
            process_audio_logic,
            file_name=file_name,
            webhook_url=webhook_url,
            job_id=job_id,
            meeting_dir=meeting_dir
        )
        return {"status": "success", "message": "Audio processing started."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

def process_audio_logic(file_name: str, webhook_url: Optional[str] = None, job_id: Optional[str] = None, meeting_dir: Optional[str] = None):
    try:
        print(json)
        start = time.time()
        audio_file_path = os.path.join('./dataset/audio', file_name)
        clean_audio = frontend_pipe.process_audio(audio_file_path, chunk_length=300, deverve=True)
        wav_file_name = audio_file_path.replace('.m4a', '.wav')
        vad_result = vad_pipe.get_vad_timestamp(clean_audio)

        diar_result, _ = diar_pipe.get_diar(wav_file_name, return_embeddings=False)
        print(f'[DEBUG] get_diar done !')
        processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)
        print(f'[DEBUG] diar preprocess done !')
        print(f'[DEBUG] non overlapped diar: {len(non_overlapped_diar)}, {non_overlapped_diar[0]}')
        chunk_emb_array = postprocess_pipe.get_chunk_emb_array(wav_file_name, non_overlapped_diar)
        print(f'[DEBUG] get chunk emb array done !')
        label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
        print(f'[DEBUG] build label mapping done !') 
        full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
        print(f'[DEBUG] apply label to full diar done !')
        final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)
        print(f'Diarization Done !: {time.time() - start}초')
        if webhook_url:
            requests.post('http://faicord-backend:8080/api/status/meetings/status-update', json={
                "statusCode": "003",
                "meetingId": meeting_dir
            })
        rttm_path = wav_file_name.replace('/audio', '/rttm').replace('.wav', '.rttm')
        diar_pipe.save_merged_rttm(final_diar, rttm_path)

        diar_result = stt_pipe.read_rttm(rttm_path)
        stt_result = stt_pipe.transcribe_by_rttm(wav_file_name, diar_result)
        print(f'STT Done !: {time.time() - start}초')
        if webhook_url:
            requests.post('http://faicord-backend:8080/api/status/meetings/status-update', json={
                "statusCode": "004",
                "meetingId": meeting_dir,
            })
        # /dataset/stt/file_name.json 
        # /dataset/stt/meeting_id/job_id.json
        stt_dir = f'./dataset/stt/{meeting_dir}'
        os.makedirs(stt_dir, exist_ok=True)
        stt_file_name = f'./dataset/stt/{meeting_dir}/{job_id}.json'
        # stt_file_name = wav_file_name.replace('/audio', '/stt').replace('.wav', '.json')
        with open(stt_file_name, "w", encoding="utf-8") as f:
            json.dump(stt_result, f, ensure_ascii=False, indent=2)

        stt_result = summary_pipe.read_stt_result(stt_file_name)
        stt_results = summary_pipe.split_stt_result(stt_result, chunk_count=3)  # [초반부, 중반부, 후반부]
        chunk_summary = '' 
        for idx in range(len(stt_results)):
            summary_result = summary_pipe.summarize(openai_summary_model, stt_results[idx], system_prompt=default_system_prompt, subrole_prompt=default_subrole_prompt) 
            chunk_summary += summary_result + '\n\n'
        total_summary = summary_pipe.summarize(openai_summary_model, chunk_summary, system_prompt=concat_system_prompt, subrole_prompt='')   
        print(f'Summarize Done !: {time.time() - start}초')

        markdown_text = summary_pipe.convert_minutes_to_markdown(total_summary)
        # save_file_name = 'faicord_' + file_name.split('.')[0] + '_summary.html'
        save_file_name = f'{job_id}.html'
        html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
        summary_dir = f'./dataset/summary/{meeting_dir}'
        os.makedirs(summary_dir, exist_ok=True)
        with open(os.path.join(summary_dir, save_file_name), "w", encoding="utf-8") as f:
            f.write(html_text)
        if webhook_url:
            requests.post(webhook_url, json={
                "status": "success",
                "job_id": job_id
                }
            )
    except Exception as e:
        logging.error(f"[process_audio_logic] Error: {e}")
        if webhook_url:
            requests.post(webhook_url, json={
                "status": "error",
                "job_id": job_id,
                "error_message": str(e)
            })

@app.post("/summarize_audio")
async def summarize_audio_endpoint(
        background_tasks: BackgroundTasks,
        meeting_dir: Optional[str] = Form(None),   # meeting_id
        system_prompt: str = Form("default"),
        subrole_prompt: str = Form("default"),
        webhook_url: Optional[str] = Form(None),
        job_id: Optional[str] = Form(None)
    ):
    try:   # uploaded_files/stt_results/meeting_dir/tmp.json 
        # 비동기 요약 작업 등록
        print(f'meeting_dir: {meeting_dir}')
        tmp_log_path = f'./dataset/stt/{meeting_dir}/tmp.json'
        with open(tmp_log_path, "r", encoding="utf-8") as f:
            meeting_log_content = json.load(f)
        
        stt_results = summary_pipe.split_stt_result(meeting_log_content, chunk_count=3)  # [초반부, 중반부, 후반부]
        # print(f'meeting_log_content: {meeting_log_content}')
        background_tasks.add_task(
            summarize_audio_logic,
            meeting_log_content=stt_results,
            meeting_dir=meeting_dir,
            user_system_prompt=system_prompt,
            user_subrole_prompt=subrole_prompt,
            webhook_url=webhook_url,
            job_id=job_id
        )
        return {"status": "success", "message": "Summarization task started."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

def summarize_audio_logic(meeting_log_content, meeting_dir, user_system_prompt, user_subrole_prompt, webhook_url, job_id):
    system_p = default_system_prompt if user_system_prompt == "default" else user_system_prompt
    subrole_p = default_subrole_prompt if user_subrole_prompt == "default" else user_subrole_prompt
    chunk_summary = '' 
    for idx in range(len(meeting_log_content)):
        summary_result = summary_pipe.summarize(openai_summary_model, meeting_log_content[idx], system_prompt=default_system_prompt, subrole_prompt=default_subrole_prompt) 
        chunk_summary += summary_result + '\n\n'
    
    # Markdown & HTML 저장
    total_summary = summary_pipe.summarize(openai_summary_model, chunk_summary, system_prompt=concat_system_prompt, subrole_prompt='')   
    markdown_text = summary_pipe.convert_minutes_to_markdown(total_summary)
    save_file_name = f"{job_id}.html"
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])

    output_path = os.path.join(f"./dataset/summary/{meeting_dir}", save_file_name)
    os.makedirs(f"./dataset/summary/{meeting_dir}", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f'Summarize Done !')
    if webhook_url:
        try:
            requests.post(webhook_url, json={
                "job_id": job_id,
                "status": "success"
            })
        except Exception as e:
            print(f"[Webhook Error] Failed to post to webhook: {e}")

@app.post("/visualize_emb")
async def visualize_emb(
        audio_files: List[UploadFile] = File(...),       # 오디오 파일 리스트
        label_list: List[str] = Form(...)                # 라벨 리스트 (Form으로 전달)
    ):
    if len(audio_files) == 0:
        raise HTTPException(status_code=400, detail="audio_files가 비어 있습니다.")
    if len(audio_files) != len(label_list):
        raise HTTPException(status_code=400, detail="파일 수와 라벨 수가 다릅니다.")

    embs = []
    file_names = []
    for file in audio_files:
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        file_names.append(file.filename)
        v = emb_pipe.get_emb_from_file(file_stream)           # -> np.ndarray shape [D]
        v = np.asarray(v, dtype=np.float32).ravel()           # 안전히 1D로
        embs.append(v)
    try:
        X = np.stack(embs, axis=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 결합 실패: {e}")
    
    xy_list = emb_pipe.get_xy_tsne(
        X,
        labels=label_list,
        file_names=file_names,
        as_dict=True,
    )
    return xy_list

@app.post("/visualize_spectogram")
async def visualize_spectogram(audio_file: UploadFile = File(...)):
    file_bytes = await audio_file.read()
    audio_stream = io.BytesIO(file_bytes)
    out = visualize_pipe.get_melspectrogram(audio_stream)
    # 프론트: <img src={"data:" + out["media_type"] + ";base64," + out["image_base64"]} />
    return out

@app.post("/visualize_waveform")
async def visualize_waveform(audio_file: UploadFile = File(...)):
    file_bytes = await audio_file.read()
    audio_stream = io.BytesIO(file_bytes)
    out = visualize_pipe.get_waveform(audio_stream)
    # 프론트: <img src={"data:" + out["media_type"] + ";base64," + out["image_base64"]} />
    return out

@app.get("/ping")
def ping():
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9050, reload=True)
