from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
from textwrap import dedent
import requests
import json
import os 

load_dotenv()
hf_api = os.getenv('HF_API')
login(token=hf_api)

url = "https://fai-rag.fingerservice.co.kr/prompt/chat"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    device="cuda",  # or "mps" on Mac
)

system_prompt_base = dedent("""
    당신은 회의 녹취록을 요약하여 구조화된 회의록 문서를 작성하는 AI 비서입니다. 사용자에게 도움이 될 수 있도록, 회의 내용을 안건별로 분류하고 논의된 핵심 사항과 결정사항, 향후 일정을 명확히 정리하세요. 출력은 마크다운 형식으로 작성합니다.
    녹취록 내 각 항목은 한 발언을 나타내며, "speaker"는 발언자 이름 (없으면 빈 문자열), "text"는 발언 내용입니다. 이 데이터를 분석하여 **내부 공유용 회의록**을 작성해 주세요:

    **요구사항:**
    - 회의에서 논의된 내용을 **주요 안건별로 분류 및 정리**합니다.
    + 각 안건 아래에 **발언자별 주요 의견**을 bullet 형태로 정리하되, 발언자는 주어진 `"speaker"` 값(SPEAKER_00 등)을 그대로 사용합니다.
    + 발언자의 실명 또는 직책은 표기하지 않습니다.
    - **결정된 사항**과 **향후 일정**(액션 아이템)이 언급된 경우 모두 식별하여 정리합니다.
    - **입력에 주어진 내용만** 사용하고, 주어진 내용에 없는 추측이나 창작을 하지 않습니다.
    - **회의 내용의 분량에 따라** 요약의 길이와 상세함을 조절합니다 (내용이 많으면 자세히, 적으면 간략히).

    **출력 형식:**
    다음의 마크다운 구조를 따라 작성하세요 (각 대괄호 부분을 실제 내용으로 채워 넣을 것):

    ## 안건
    1. [안건 또는 주제 1]
    2. [안건 또는 주제 2]
    ...

    ## 논의 사항
    ### 안건 1: [안건 1의 제목]
    - [발언자1 이름 또는 '참석자']: [해당 발언자의 핵심 의견 요약]
    - [발언자2 이름 또는 '참석자']: [다른 참여자의 주요 발언 요약]
    ... (필요한 만큼 추가)
    - 결정사항: [안건 1과 관련하여 결정된 사항이 있다면 명시]

    ### 안건 2: [안건 2의 제목]
    - [발언자]: [핵심 의견 요약]
    ...
    - 결정사항: [결정된 사항이 있으면 명시, 없으면 제외]

    ## 참고사항
    ### 향후일정
    - [향후 수행해야 할 작업 또는 일정 1]: [예정 일자 또는 기한]
    - [향후 수행해야 할 작업 또는 일정 2]: [예정 일자 또는 기한]
    ... (필요 시 추가)

    ### 결정사항
    1. [회의 전체적으로 나온 중요한 결정사항 1]
    2. [결정사항 2]
    ... (필요 시 추가)

    위 형식을 참고하여, 회의 내용을 **안건별**로 요약하고 **결정사항**과 **향후일정**까지 빠짐없이 정리해 주세요.
""").strip()

sub_guidelines = dedent("""
    - 각 안건의 논의 사항은 발언자의 핵심 발언을 **요약하되**, 세부 설명, 구현 방식, 예시, 수치 등이 포함되도록 작성해 주세요.
    - 기술, 기능, 구조, 적용 방식 등은 **단순 언급이 아닌 설명 중심으로** 정리해 주세요.  
    - 동일 화자가 여러 주제를 언급한 경우, **항목별로 문단을 나눠 상세하게 정리**해 주세요.
    - 시연 관련 발언은 "**기능 설명 + 실제 시연 흐름 + 결과 또는 효과**" 순서로 서술해 주세요.  
    - 논의 중 등장한 **핵심 기술, 기능, 구조적 요소**는 안건 내에서 **별도 항목 또는 문단**으로 구분하여 설명해 주세요.  
    - 해당 요소에 대해 가능한 한 **적용 방식, 구성 조건, 수치 범위, 제약사항 및 실제 사례** 등을 포함해 주세요.
        ex. 사용자 정의 기능, 인식 정확도 보정 방식, 학습 방법론, 인프라 운영 구조 등            
    - 발언자 이름은 JSON의 "speaker" 필드 값 그대로 사용합니다 (예: SPEAKER_00, SPEAKER_01 등). 실명, 직책 등은 사용하지 않습니다.
    - 회의 발언 수가 많은 경우, 안건당 **최소 5개 이상의 bullet 또는 문단**을 유지해 주세요.
    - 요약은 단순한 축약이 아니라, **발언 흐름과 논리 구조를 재구성하여 명확하게 전달하는 것**을 목표로 합니다.
    - 기술적 설명, 실제 시연, 운영 구조, 적용 사례, 협업/테스트 계획 등 **성격이 다른 논의는 반드시 별도의 안건으로 구분**해 주세요.
    - 발언 내용이 반복되거나 유사한 경우, **중복된 문장은 생략하거나 하나로 통합하여 핵심만 정리**해 주세요.
    - 동일한 주제를 여러 번 반복한 경우, **"강조 발언"인지 여부를 판단**해 실제 의미 변화가 없다면 **1회만 요약**해 주세요.
    - 반복된 표현으로 인해 **정보 밀도나 문서 품질이 떨어지지 않도록** 정리해 주세요.
    - 화자의 발언 중 **일정/타이밍/회의 시점 관련 표현**(예: '다음 주', '이번 달 말', 'POC 끝나고 나서', '다음 회의 때 다시 보기로')은 반드시 "향후 일정" 섹션에 정리해 주세요.
        - 날짜가 불명확해도 **상대적 표현**(다음 회의, 시연 완료 후 등)을 일정 항목으로 포함해 주세요.
        - 다음 회의 확정이 없어도 “일정 조율 예정”, “시연 이후 논의 예정” 등으로 명시해 주세요.
    - **제품화 제안, 실사용 경험, 경쟁력 차별화 의견** 등 비정형 발언도 **중요 항목으로 누락 없이** 요약해 주세요.
    - **어떤 내용도 화자를 바꾸거나 대표화하지 마세요.** (예: SPEAKER_06 발언을 SPEAKER_00에게 합치지 않기)
""").strip()

system_prompt = f"{system_prompt_base}\n\n[추가 지침]\n{sub_guidelines}"

def transcript_to_str(transcript, mode="json"):
    """ transcript(list[dict]) -> str """
    if mode == "json":
        return "```json\n" + json.dumps(transcript, ensure_ascii=False, indent=2) + "\n```"
    elif mode == "lines":
        return "\n".join(f'{t.get("speaker","")}: {t.get("text","")}' for t in transcript)
    else:
        raise ValueError("mode must be 'json' or 'lines'")

with open("./dataset/stt/faicord_20241220.json", "r", encoding="utf-8") as f:
    stt_result = json.load(f)

# stt_result가 이미 [{"speaker","text"}...] 형태라고 하셨으니 그대로 사용
user_content = (
    "다음 녹취록을 위 지침에 따라 요약해 마크다운으로 작성해 주세요.\n\n"
    + transcript_to_str(stt_result, mode="json")
)

# =========================
# 3) API 요청 페이로드
# =========================
payload = {
    "query": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ],
    "stream": False
}

# 여러분의 실제 엔드포인트로 교체하세요.
url = "https://fai-rag.fingerservice.co.kr/prompt/chat"

headers = {
    "Content-Type": "application/json; charset=utf-8",
    # "Authorization": "Bearer <YOUR_KEY>",  # 필요 시
}

# =========================
# 4) 전송 & 응답 처리
# =========================
try:
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 서버 스펙에 따라 응답 필드가 다를 수 있으니 안전하게 처리
    content = None
    if isinstance(data, dict):
        if "response" in data:
            content = data["response"]
        elif "choices" in data and data["choices"]:
            # OpenAI류 포맷 가정
            msg = data["choices"][0].get("message", {})
            content = msg.get("content")
        elif "output" in data:
            content = data["output"]

    if not content:
        # 혹시 다른 키를 쓰는 서버라면 전체를 출력해 디버깅에 도움
        print("응답 본문 키를 찾지 못했습니다. 원본 JSON:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(content)

except requests.HTTPError as e:
    # 500 등 서버 에러 시, 보내는 페이로드 길이와 앞부분 로그
    print(f"HTTPError {resp.status_code}: {resp.text[:500]}")
    print(f"[디버그] payload size: {len(json.dumps(payload, ensure_ascii=False))} bytes")
except Exception as e:
    print("요청 중 예외 발생:", repr(e))