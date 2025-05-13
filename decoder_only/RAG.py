import os
import sys
import openai
import time
import pandas as pd
import nest_asyncio
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from decoder_only.urls import Enterprise_URLS
import nest_asyncio
from transformers import AutoTokenizer, pipeline
import torch
nest_asyncio.apply()
sys.path.append('../..')


class MITREAnalysis:
    def __init__(self, api_key, data_source=None, mode='url', llm_model_name="gpt-3.5-turbo-1106"):
        self.llm_model_name = llm_model_name
        if "gpt" in llm_model_name:
            self.setup_openai(api_key)
        self.data = self.load_data(data_source, mode)      
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectordb = FAISS.from_documents(documents=self.data, embedding=self.embeddings)
        self.llm = self.setup_llm(llm_model_name, api_key)
        self.prompt_template = self.build_qa_chain_prompt()

    def setup_openai(self, api_key):
        openai.api_key = api_key

    def setup_llm(self, llm_model_name, api_key):
        if "gpt" in llm_model_name:
            return ChatOpenAI(model_name=llm_model_name, temperature=0, openai_api_key=api_key, model_kwargs={"seed": 1106})
        elif "llama" in llm_model_name.lower() or "meta" in llm_model_name.lower():
            # Create HuggingFace pipeline
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=api_key)
            model = pipeline(
                "text-generation",
                model=llm_model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                max_length=1024,
                do_sample=True,
                temperature=0.01,
                top_k=20,
                token=api_key
            )
            llm = HuggingFacePipeline(pipeline=model)
            return llm
        else:
            raise ValueError(f"Unsupported model: {llm_model_name}")

    def load_data(self, data_source, mode):
        if mode == 'csv':
            # Phiên bản mới của CSVLoader - thay đổi cách sử dụng
            self.loader = CSVLoader(
                file_path=data_source,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'encoding': 'ISO-8859-1'
                },
                source_column='Procedures'
            )
            
            # Đọc dữ liệu
            documents = self.loader.load()
            
            # Đọc URL từ CSV và thêm vào metadata
            df = pd.read_csv(data_source, encoding='ISO-8859-1')
            for i, doc in enumerate(documents):
                if i < len(df):
                    doc.metadata['URL'] = df.loc[i, 'URL']
            
            self.data = documents
        elif mode == 'all_urls':
            self.data = self.load_and_split_web_content_all(Enterprise_URLS)
        elif mode == 'reference_url':
            urls = data_source
            self.data = self.load_and_split_web_content(urls)
        elif mode == 'similar_procedure_urls':
            urls = data_source
            self.data = self.load_and_split_web_content(urls)
        else:
            raise ValueError("Invalid mode.")
        return self.data
     
    def perform_procedure_retrieval(self, procedure, url, tactics, k=3):
        docs = self.vectordb.similarity_search(procedure, k=k)
        retr_procecdures = []
        retr_urls = []
        for doc in docs[1:]:
            retr_procecdures.append(doc.metadata.get('source'))
            retr_urls.append(doc.metadata.get('URL'))
    
        procedure_data = {
            "Procedure": procedure,
            "Procedure URL": url, 
            "Retrieved Procedures": retr_procecdures, 
            "Retrieved Procedure URLs": retr_urls,
            "Tactic(s)": tactics,
            
        }
        return procedure_data
    
    def perform_similarity_search(self, question, k=3):
        docs = self.vectordb.similarity_search(question, k=k)
        for doc in docs:
            print(doc.metadata)
            

    def build_qa_chain_prompt(self):
        template = """You are a cybersecurity analyst with the expertise in analyzing cyberattack procedures. Consider the relevant context provided below and answer the question.

        Relevant Context: {context}

        Question: {question}

        Please write the response in the following format: tactic(s)
        """
        return PromptTemplate.from_template(template)


    def perform_qa(self, question, prompt_template):
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        result = qa_chain({"query": question})
        print(result["result"],'\n')
        return result


    def load_questions_from_csv(self, csv_file):
        list_of_questions = []
        df = pd.read_csv(csv_file)
        for procedure in df['Procedures']:
            temp = f"Knowing that <<{procedure}>>, what MITRE ATT&CK tactics will a cyber adversary achieve with this technique?"
            self.perform_similarity_search(temp, k=3)
            print('------------------')
            list_of_questions.append(temp)
        return list_of_questions


    def perform_qa_for_list(self, list_of_questions):
        predictions = []
        for question in list_of_questions:
            while True:
                try:
                    print(question)
                    result = self.perform_qa(question, self.prompt_template)
                    predictions.append(result)
                    break
                except (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout,
                        openai.error.OpenAIError, openai.error.ServiceUnavailableError):
                    time.sleep(5)
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(5)
        return predictions


    def load_and_split_web_content(self, url):
        docs = WebBaseLoader(url).load()
         
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 5000,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        return splits 

    
    def load_and_split_web_content_all(self, urls):
        loader = WebBaseLoader(urls)
        loader.requests_per_second = 1
        docs = loader.aload()


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 5000,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print("chunks:", len(splits))
        return splits