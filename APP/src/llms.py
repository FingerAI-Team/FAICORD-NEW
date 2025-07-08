from openai import OpenAI
import torch

class LLMModel():
    def __init__(self, config):
        self.config = config 

    def set_gpu(self, model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"    
        model.to(self.device)
    
    def set_generation_config(self, max_tokens=3000, temperature=0.0):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

class LLMOpenAI(LLMModel):
    def __init__(self, config, api_key):
        super().__init__(config)
        self.client = OpenAI(api_key=api_key)

    def set_generation_config(self):
        self.gen_config = {
            "max_tokens": self.config['max_tokens'],
            "temperature": self.config['temperature']
        }

    def set_summary_guideline(self):
        '''
        STT 결과값을 보고, 회의록을 작성하기 위한 요약 템플릿입니다.  
        '''
        self.system_role = """
        당신은 회의 녹취록을 요약하여 구조화된 회의록 문서를 작성하는 AI 비서입니다. 사용자에게 도움이 될 수 있도록, 회의 내용을 안건별로 분류하고 논의된 핵심 사항과 결정사항, 향후 일정을 명확히 정리하세요. 출력은 마크다운 형식으로 작성합니다.
        녹취록 내 각 항목은 한 발언을 나타내며, "speaker"는 발언자 이름 (없으면 빈 문자열), "text"는 발언 내용입니다. 이 데이터를 분석하여 **내부 공유용 회의록**을 작성해 주세요:
        
        **요구사항:**
        - 회의에서 논의된 내용을 **주요 안건별로 분류 및 정리**합니다.
        - 각 안건 아래에 **발언자별 주요 의견**을 bullet 형태로 정리합니다. 화자 이름이 없으면 "참석자" 등으로 표기합니다.
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
        """

        self.sub_role = """
        회의록 예시는 다음과 같아. 주요 안건들을 전사 기록에서 도출하고, 주요 안건별 각 화자들이 어떤 발언과 주장을 했는지 구체적으로 정리해서 작성해줘. 
        - 주요 안건
            1. 외부 AI 코딩 도구 비교 및 도입 검토 
            2. Figma 기반 자동화 개발 프로세스 검증 (PoC)
            3. 자체 sLLM 기반 시스템 구축 방향 논의 
            4. 향후 AI 생산성 도구 내재화 및 사업화 전략 논의 
        - 논의 사항 
            1. 외부 AI 코딩 도구 비교 및 도입 검토
                - SPEAKER_00: 커서(Cursor)와 클로드 코드(Claude Code)를 중심으로 성능, 특징, 생산성 등을 비교함. 커서는 UI 중심 프론트엔드 개발에 강점을 보이며, 
                              클로드 코드는 대규모 백엔드 리팩토링, 자동화에 유리하다고 설명
                - SPEAKER_01: 커서와 클로드 활용 경험을 바탕으로, 커서를 활용한 생산성 향상이 체감된다는 의견 제시. 실제 사례에서 신규 개발자의 온보딩 속도가 빨라졌음을 언급 
                - SPEAKER_02: 클로드나 ChatGPT 기반 툴도 무작정 도입하는 것보다, sLLM 기반 내재화 필요성에 대해 강조. 향후 적용 사업을 고려했을 때, 외부 도구 의존은 한계가 존재함
                * 결정 사항: 현재는 커서를 우선적으로 활용하고, 클로드와의 비교 검증도 병행. 커서 사용을 통한 개발자 온보딩 및 프론트엔드 개발 효율성 검토    
            2. Figma 기반 자동화 개발 프로세스 검증 (PoC)
                - SPEAKER_00: Figma와 커서를 연동하여 코드 제너레이션 가능성을 실험중. MCP(Middleware Communication Protocol)를 통해 자동화 가능성이 있고, 향후 PoC로 구체화할 예정
                - SPEAKER_01: 기존 Figma 디자인을 활용하여 개발 속도 향상 가능성에 기대. 실질적으로 코드가 얼마나 자동 생성되는지 검증이 중요함을 강조함
                - SPEAKER_02: PoC 진행을 위해 타겟 프로젝트 (파로스 또는 신규 개발 프로젝트) 선정 필요에 대해 언급 
                * 결정 사항: SPEAKER_00이 Figma-커서 연동 PoC 구성 및 일정 수립을 맡음. 다음 회의에서 PoC 결과를 공유하고, 단계별 적용 가능성을 검토하기로 함 
            3. 자체 sLLM 기반 시스템 구축 방향 논의 
                - SPEAKER_00: 벡터 DB를 기반으로 한 커스터마이징형 sLLM 시스템 구조 설명, 프로젝트 소스를 벡터화하고, 이를 기반으로 자동 코드 생성 흐름 제시 
                - SPEAKER_01: 온프레밋 기반으로 내재화 가능한 구조 제안. AI 생산성 도구 내재화는 검증 가능한 데이터 축적이 핵심임을 강조. 
                - SPEAKER_03: 현재 GPU 서버등의 사양상, 고성능 LLM 실행에 제약이 존재하고, DeepSeek Coder 등 33B 모델 활용 시 메모리 부족 가능성에 대해 지적
                * 결정 사항: 시범적으로 파로스 또는 M&A 플랫폼 개발 프로젝트를 대상으로 sLLM 검증 환경 구성 추진 
        - 참고 사항 (향후 일정)
            - PoC 일정 정리: SPEAKER_00이 Figma-커서 연동 PoC를 다음 주까지 완료하고, 결과를 공유하기로 함
            - 회의 일정 변경: 매주 월요일 -> 매주 화요일 오후 1시 30분으로 변경 
        - 결정 사항
            1. 커서 중심으로 Figma 연동 자동화 개발 PoC 우선 추진 
            2. 클로드 코드 및 기타 도구는 비교 검토하며 필요 시 병렬 테스트 
            3. 자체 sLLM + 벡터DB 기반 시스템 구조 시범 구성 시작 
            4. AI팀은 환경 구성 및 가이드 설계, 서비스팀은 실제 개발 적용 
            5. TFT 1기 목표는 생산성 향상 사례 및 표준 프로세스 정립  
        """

    def set_prompt_template(self, query):
        self.meeting_log_prompt_template = """
        {query} 
        """
        return self.meeting_log_prompt_template.format(query=query)
        
    def get_response(self, query, role="너는 금융권에서 일하고 있는 조수로, 사용자 질문에 대해 간단 명료하게 답을 해주면 돼", sub_role="", model='gpt-4o'):
        try:
            sub_role = sub_role
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "system", "content": sub_role},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.gen_config['max_tokens'],
                temperature=self.gen_config['temperature'],
            )
        except Exception as e:
            return f"Error: {str(e)}"
        return response.choices[0].message.content