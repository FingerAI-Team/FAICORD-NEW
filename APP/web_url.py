from src import FrontendPipe, VADPipe, DIARPipe, PostProcessPipe, STTPipe, SummaryPipe
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional
import markdown
import logging 
import uvicorn
import time
import json
import os

load_dotenv()

app = FastAPI()
file_path = os.getenv('FILE_PATH', './dataset/audio/')
vad_config = os.path.join('./models', 'pyannote_vad_config.yaml')
diar_config = os.path.join('./models', 'pyannote_diarization_config.yaml')

with open('./config/generation_config.json') as f:
    generation_config = json.load(f)

with open('./config/default_system_prompt.txt', "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open('./config/default_subrole_prompt.txt', "r", encoding="utf-8") as f:
    subrole_prompt = f.read()

frontend_pipe = FrontendPipe()
vad_pipe = VADPipe(vad_config)
diar_pipe = DIARPipe(diar_config)
postprocess_pipe = PostProcessPipe()

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
    audio_file_path = './dataset/audio/' + file_name 
    try:
        # Background task 등록
        background_tasks.add_task(
            process_audio_logic,
            file_path=audio_file_path,
            webhook_url=webhook_url,
            job_id=job_id
        )
        return {"status": "success", "message": "Audio processing started."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


def process_audio_logic(file_path: str, webhook_url: Optional[str] = None, job_id: Optional[str] = None):
    try:
        start = time.time()
        clean_audio = frontend_pipe.process_audio(file_path, chunk_length=300, deverve=True)
        vad_result = vad_pipe.get_vad_timestamp(clean_audio)

        diar_result, _ = diar_pipe.get_diar(file_path, return_embeddings=False)
        processed_diar, non_overlapped_diar = diar_pipe.preprocess_result(diar_result=diar_result, vad_result=vad_result)
        chunk_emb_array = postprocess_pipe.get_chunk_emb_array(file_path, non_overlapped_diar)
        label_mapping_dict = postprocess_pipe.build_label_mapping_dict(chunk_emb_array)
        full_diar = postprocess_pipe.apply_labels_to_full_diar(processed_diar, non_overlapped_diar)
        final_diar = postprocess_pipe.apply_label_mapping_to_diar(full_diar, label_mapping_dict)
        print(f'Diarization Done !: {time.time() - start}초')
        if webhook_url:
            requests.post(webhook_url, json={
                "status": "success",
                "job_id": job_id
            })
        rttm_path = file_path.replace('.wav', '.rttm')
        diar_pipe.save_merged_rttm(final_diar, os.path.join('./dataset/rttm/', os.path.basename(rttm_path)))

        diar_result = stt_pipe.read_rttm(rttm_path)
        stt_result = stt_pipe.transcribe_by_rttm(file_path, diar_result)
        print(f'STT Done !: {time.time() - start}초')
        if webhook_url:
            requests.post(webhook_url, json={
                "status": "success",
                "job_id": job_id,
            })
        stt_file_name = os.path.basename(file_path).replace('.wav', '.json')
        with open(os.path.join('./dataset/stt/', stt_file_name), "w", encoding="utf-8") as f:
            json.dump(stt_result, f, ensure_ascii=False, indent=2)

        stt_result = summary_pipe.read_stt_result(os.path.join('./dataset/stt/stt_file_name'))
        summary_result = summary_pipe.summarize(openai_summary_model, stt_result)
        print(f'Summarize Done !: {time.time() - start}초')

        markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
        save_file_name = 'faicord_' + os.path.basename(file_path).split('.')[0] + '_summary.html'
        html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
        with open(os.path.join('./dataset/summary/', save_file_name), "w", encoding="utf-8") as f:
            f.write(html_text)
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
        tmp_log_path = f'./dataset/stt/{meeting_dir}/tmp.json'
        with open(meeting_log_path, "r", encoding="utf-8") as f:
            meeting_log_content = json.load(f)

        background_tasks.add_task(
            summarize_audio_logic,
            meeting_log_content=meeting_log_content,
            user_system_prompt=system_prompt,
            user_subrole_prompt=subrole_prompt,
            webhook_url=webhook_url,
            job_id=job_id
        )
        return {"status": "success", "message": "Summarization task started."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

def summarize_audio_logic(meeting_log_content, user_system_prompt, user_subrole_prompt, webhook_url, job_id):
    system_p = default_system_prompt if user_system_prompt == "default" else user_system_prompt
    subrole_p = default_subrole_prompt if user_subrole_prompt == "default" else user_subrole_prompt
    summary_result = summary_pipe.summarize(
        openai_summary_model,
        meeting_log_content,
        system_prompt=system_p,
        subrole_prompt=subrole_p
    )
    # Markdown & HTML 저장
    markdown_text = summary_pipe.convert_minutes_to_markdown(summary_result)
    base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    save_file_name = f"faicord_{base_name}_summary.html"
    html_text = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])

    output_path = os.path.join("./dataset/summary/", save_file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    if webhook_url:
        try:
            requests.post(webhook_url, json={
                "job_id": job_id,
                "status": "completed"
            })
        except Exception as e:
            print(f"[Webhook Error] Failed to post to webhook: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)