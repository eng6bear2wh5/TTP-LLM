import openai
import time
import pandas as pd
import random
from transformers import AutoTokenizer
import transformers
from transformers import pipeline
import torch
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

def get_completion(prompt, model="gpt-4-1106-preview"):
    if "gpt" in model:
        api_key = config.get('API', 'OpenAI_Key')
        openai.api_key = api_key
        messages = [{"role": "system", 
                    "content":"You are a cybersecurity analyst with the expertise in analyzing cyberattack procedures."},
                    {"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            seed=1106,
        )
        result = response.choices[0].message["content"]
    elif "llama" in model:
        model = "meta-llama/Llama-2-7b-chat-hf" 
        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
        llama_pipeline = pipeline(
                    "text-generation",  # LLM task
                    model=model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        sequences = llama_pipeline(
                    prompt,
                    do_sample=True,
                    top_k=20,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=1024,
                )
        result = sequences[0]['generated_text']
    return result


def load_questions_from_csv(csv_file):
    list_of_questions = []
    df = pd.read_csv(csv_file)
    for procedure in df['Description']:
        temp = f"Knowing that <<{procedure}>>, what MITRE ATT&CK tactics will a cyber adversary achieve with this technique?"
        list_of_questions.append(temp)
    return list_of_questions


def prediction(list_of_questions, model="gpt-4-1106-preview"):
    predictions = []
    counter = 0
    for question in list_of_questions:
        counter += 1
        print('Procedure:', counter)
        prompt = f"""{question}

        Please write the response in the following format: tactic(s)
        """
        while True:
            try:
                print(question)
                result = get_completion(prompt, model=model)
                print(result,'\n')
                predictions.append(result)
                break
            except (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout,
                    openai.error.OpenAIError, openai.error.ServiceUnavailableError):
                delay = random.randint(2, 6)
                time.sleep(delay)
    return predictions

