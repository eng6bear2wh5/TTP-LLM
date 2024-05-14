#!pip install -q langchain
#!pip install -q Pydantic==1.10.12
#!pip install -q chromadb
#!pip install -q tiktoken
#!pip install -q lark
#!pip install faiss-cpu


import os
import sys
import openai
import time
import pandas as pd
import nest_asyncio
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, NLTKTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from decoder_only.urls import Enterprise_URLS
import nest_asyncio
nest_asyncio.apply()
sys.path.append('../..')


class MITREAnalysis:
    def __init__(self, api_key, data_source=None, mode='url', llm_model_name="gpt-3.5-turbo-1106"):
        self.setup_openai(api_key)
        self.data = self.load_data(data_source, mode)      
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectordb = FAISS.from_documents(documents=self.data, embedding=self.embeddings)
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0, openai_api_key=api_key, model_kwargs={"seed": 1106})
        self.prompt_template = self.build_qa_chain_prompt()

    def setup_openai(self, api_key):
        openai.api_key = api_key

    def load_data(self, data_source, mode):
        if mode == 'csv':
            self.loader = CSVLoader(data_source, source_column='Procedures', metadata_columns=['URL'] , encoding="ISO-8859-1")
            self.data = self.loader.load()
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