import argparse
import configparser
import openai
from decoder_only.RAG import MITREAnalysis
from decoder_only.prompt_only import *
import pandas as pd
import random
import time
import ast
from decoder_only.urls import *
from encoder_only import *

def handle_nan(value):
    if isinstance(value, list):
        return value  
    if pd.isna(value):
        return None 
    try:
        return ast.literal_eval(value) 
    except ValueError:
        return None 
    
def main():
    if args.type=='encoder_only':
        print('Go over to the encoder_only folder and run the notebooks')
        #preprocess
        #train
        #evaluate
    elif args.type=='decoder_only':
        if 'gpt' in args.llm:
            api_key = config.get('API', 'OpenAI_Key')
        else:
            api_key = config.get('API', 'HuggingFace_Key')
        dataset_path = './data/MITRE_Procedures.csv'
        df = pd.read_csv(dataset_path)

        if args.mode=='prompt_only':
            list_of_questions = load_questions_from_csv(dataset_path)
            predictions = prediction(list_of_questions, args.llm)

        if args.mode=='all_urls':
            analysis = MITREAnalysis(api_key, data_source=None,  mode=args.mode, llm_model_name=args.llm)
            list_of_questions = analysis.load_questions_from_csv(dataset_path)
            predictions = analysis.perform_qa_for_list(list_of_questions)

        if args.mode=='reference_url':
            predictions = []
            counter=0
            for procedure, url in zip(df['Procedures'], df['URL']):
                counter += 1
                print('Procedure:', counter)
                analysis = MITREAnalysis(api_key=api_key, data_source=url, mode=args.mode, llm_model_name= args.llm)
                prompt_template = analysis.build_qa_chain_prompt()
                while True:
                    try:
                        question = f"Knowing that <<{procedure}>>, what MITRE ATT&CK tactics will a cyber adversary achieve with this technique?"
                        print(question)
                        analysis.perform_similarity_search(question, k=3)
                        print('------------------')
                        result = analysis.perform_qa(question, prompt_template)
                        print('------------------')
                        predictions.append(result)
                        break
                    except (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout,
                            openai.error.OpenAIError, openai.error.ServiceUnavailableError):
                        time.sleep(8)

        if args.mode=='similar_procedure_urls':
            predictions = []
            analysis = MITREAnalysis(api_key=api_key, data_source=dataset_path, mode='csv', llm_model_name=args.llm)
            all_procedures_df = pd.DataFrame(columns=["Procedure", "Procedure URL", "Retrieved Procedures", "Retrieved Procedure URLs", "Tactic(s)"])
            for procedure, url, tactic1, tactic2 in zip(df['Description'], df['URL'], df['Tactic1'], df['Tactic2']):
                procedure_data = analysis.perform_procedure_retrieval(procedure, url, [tactic1, tactic2])
                procedure_data_df = pd.DataFrame([procedure_data])
                all_procedures_df = pd.concat([all_procedures_df, procedure_data_df], ignore_index=True)
            
            all_procedures_df['Tactic(s)'] = all_procedures_df['Tactic(s)'].apply(handle_nan)

            max_len = all_procedures_df['Tactic(s)'].str.len().max()
            for i in range(max_len):
                column_name = f'Tactic{i+1}' 
                all_procedures_df[column_name] = all_procedures_df['Tactic(s)'].apply(lambda x: x[i] if i < len(x) and x[i] is not None else None)
            
            all_procedures_df.drop('Tactic(s)', axis=1, inplace=True)
            all_procedures_df.to_csv('./data/Procedures_RAG_Similarity.csv', index=False)
            all_procedures_df = pd.read_csv('./data/Procedures_RAG_Similarity.csv')
            all_procedures_df['Retrieved Procedure URLs'] = all_procedures_df['Retrieved Procedure URLs'].apply(lambda x: x.replace("'", '"'))
            all_procedures_df['Retrieved Procedure URLs'] = all_procedures_df['Retrieved Procedure URLs'].apply(lambda x: ast.literal_eval(x))
            counter = 0
            for procedure, retrieved_urls in list(zip(all_procedures_df['Procedures'], all_procedures_df['Retrieved Procedure URLs'])):
                counter += 1
                print('Procedure:', counter)
                analysis = MITREAnalysis(api_key= api_key, data_source= retrieved_urls, mode=args.mode, llm_model_name=args.llm)
                prompt_template = analysis.build_qa_chain_prompt()
                while True:
                    try:
                        question = f"Knowing that <<{procedure}>>, what MITRE ATT&CK tactics will a cyber adversary achieve with this technique?"
                        print(question)
                        result = analysis.perform_qa(question, prompt_template)
                        print('------------------')
                        predictions.append(result)
                        break
                    except (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout,
                            openai.error.OpenAIError, openai.error.ServiceUnavailableError):
                        delay = random.randint(2, 6)
                        time.sleep(delay)

        final_df = pd.DataFrame(predictions)    
        final_df.to_csv(f'./results/preds_{args.llm}_{args.mode}.csv', index=False)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    parser = argparse.ArgumentParser(description="MITRE Analysis Tool")
    parser.add_argument(
        '--type', 
        type=str, 
        default='decoder_only', 
        help='encoder_only or decoder_only')
    parser.add_argument(
        '--llm', 
        type=str, 
        default='gpt-3.5-turbo-1106', 
        help='LLM for use: e.g., gpt-3.5-turbo')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='prompt_only', 
        help='Mode of operation (prompt_only, reference_url, similar_procedure_urls, all_urls)')
    args = parser.parse_args()

    main()
    # TO RUN: python main.py --type decoder_only --mode prompt_only --llm gpt-3.5-turbo
   
