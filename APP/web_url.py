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
import time
import json
import os

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
        processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)
        chunk_emb_array = postprocess_pipe.get_chunk_emb_array(wav_file_name, non_overlapped_diar)
        label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
        full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
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
    total_summary = summary_pipe.summarize(openai_summary_model, chunk_summary, system_prompt=default_system_prompt, subrole_prompt='')   
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
def visualize_emb(req: VisualizeEmbRequest):
    '''
    audio_file_list = ['speaker1.wav', 'speaker2.wav', 'speaker3.wav', ... 'speakerN.wav']
    lable_list = ['SPEAKER_00', 'SPEAKER_00', 'SPEAKER_01', ... , 'SPEAKER_N']
    '''
    audio_file_list = req.audio_file_list
    speaker_label_list = req.label_list
    if len(audio_file_list) == 0:
        raise HTTPException(status_code=400, detail="audio_file_list가 비어 있습니다.")
    if len(audio_file_list) != len(speaker_label_list):
        raise HTTPException(status_code=400, detail="audio_file_list와 label_list 길이가 다릅니다.")

    missing = [p for p in audio_file_list if not os.path.isfile(p)]
    if missing:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {missing[:3]}{' ...' if len(missing)>3 else ''}")
    
    embs = []
    for p in audio_file_list:
        v = emb_pipe.get_emb_from_file(p)           # -> np.ndarray shape [D]
        v = np.asarray(v, dtype=np.float32).ravel() # 안전히 1D로
        embs.append(v)
    X = np.stack(embs, axis=0)
    xy_list = emb_pipe.get_xy_tsne(
        X,
        labels=speaker_label_list,
        file_names=[os.path.basename(f) for f in audio_file_list],
        as_dict=True,           # ← 파일/라벨까지 포함한 JSON 레코드로
        metric="cosine",
        seed=42
    )
    return xy_list

@app.post("/visualize_spectogram")
def visualize_spectogram(req: MelReq):
    if not os.path.isfile(req.audio_file):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    out = visualize_pipe.get_melspectrogram(req.audio_file)
    # 프론트: <img src={"data:" + out["media_type"] + ";base64," + out["image_base64"]} />
    return out

@app.post("/visualize_waveform")
def visualize_waveform(req: WaveformReq):
    if not os.path.isfile(req.audio_file):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    out = visualize_pipe.get_waveform(req.audio_file)
    # 프론트: <img src={"data:" + out["media_type"] + ";base64," + out["image_base64"]} />
    return out

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9050, reload=True)