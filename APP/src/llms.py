from openai import OpenAI
import torch

class LLMModel():
    def __init__(self, config):
        self.config = config 

    def set_gpu(self, model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"    
        model.to(self.device)
    
    def set_generation_config(self, max_tokens=15000, temperature=0.05):
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

    def set_summary_guideline(self, system_prompt, subrole_prompt):
        '''
        STT 결과값을 보고, 회의록을 작성하기 위한 요약 템플릿입니다.  
        '''
        self.system_role = system_prompt
        self.sub_role = subrole_prompt

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