from openai import OpenAI
from langsmith import wrappers
from langsmith import traceable
import os


class LLM:
    def __init__(self, openai_model_name='gpt-3.5-turbo-0125'):
        self.llm = wrappers.wrap_openai(OpenAI(api_key=""))
        self.openai_model_name = openai_model_name


    def response_parser(self, response):
        return response.choices[0].message.content
    
    def initialise_llm(self, api_key):
        self.llm = wrappers.wrap_openai(OpenAI(api_key=api_key))

    @traceable
    def step(self, prompt):
        response =  self.llm.chat.completions.create(
            model=self.openai_model_name,
            messages=[
                {"role": "user", "content": prompt}],
        )
        response = self.response_parser(response=response)
        return response
        
