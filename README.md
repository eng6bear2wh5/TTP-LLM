# TTP-LLM
Advancing TTP Analysis: Harnessing the Power of Large Language Models with Retrieval Augmented Generation

Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise and complex dependencies. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. It is, however, unclear how LLMs can be used in an efficient and proper way to provide accurate responses for critical domain such as cybersecurity. This leads us to investigate how to better use the two types of LLMs: small-scale encoder-only (e.g., RoBERTa) and larger decoder-only (e.g., GPT-3.5) LLMs to comprehend and summarize TTPs with the intended purposes (i.e., tactics) of a cyberattack procedure. This work studies and compares the uses of supervised fine-tuning (SFT) of encoder-only LLMs vs. Retrieval Augmented Generation (RAG) for decoder-only LLMs (without fine-tuning). Both SFT and RAG techniques presumably enhance the LLMs with  relevant contexts for each cyberattack procedure. Our studies show decoder-only LLMs with RAG achieves better performance than encoder-only models with SFT, particularly when directly relevant context is extracted by RAG. The decoder-only results could suffer low "precision" while achieving high "recall". Our findings further highlight a counter-intuitive observation that more generic prompts tend to yield better predictions of cyberattack tactics than those that are more specifically tailored.

## Setup
Create a conda environment and install the libraries:
```python
conda create --name TTP-LLM python=3.10
conda activate TTP-LLM
pip install -r requirements.txt
```

## Usage

In the data folder, there are four files:
1) "MITRE_Tactic_and_Techniques_Descriptions.csv" is the training dataset (crawled from the MITRE ATT&CK framework) and is used for training encoder_only models to understand and classify tactics.
2) "MITRE_Procedures.csv" is the main dataset that is crawled from the MITRE ATT&CK framework that contains the procedures' descriptions along with their corresponding tactic(s)
3) "MITRE_Procedures_encoded.csv" is the encoded version of the dataset for evaluation
4) "Procedures_RAG_Similarity.csv" is the file containing the top-2 most similar procedures along with their tactics

The encoder_only folder contains the notebooks on fine-tuning the encoder_only models (e.g., roberta and secureBERT) with "MITRE_Tactic_and_Techniques_Descriptions.csv" data and evaluating them on "MITRE_Procedures.csv".

The decoder_only folder contains .py files on how to use decoder_only LLMs with or without RAG techniques.
The "prompt_only.py" file is the process of directly accessing the OpenAI GPT models with prompt engineering. 

The "RAG.py" file contains three different RAG techniques:
1) all_urls: This is to load all the ICS URLs from MITRE ATT&CK for retrieval.
2) reference_url: This is to load the reference URL of each specific procedure description (the reference URL is in the ICS_Procedures_main.csv dataset).
3) similar_procedure_urls: This is to retrieve URLs that correspond to the top-2 'target' procedure descriptions, which are most similar to the 'source' procedure specified in the query.

## How to Run

1) Create a "config.ini" file and put your API key in the following format:
```python
[API]
OpenAI_Key = <YOUR_API_KEY>
HuggingFace_Key = <YOUR_API_KEY>

```
2) Run the "main.py" file with the following line:
```python
python main.py --mode [prompt_only, reference_url, similar_procedure_urls, all_urls] --llm [LLM]
```
Choose one of the four modes in --mode (e.g., --mode all_urls) and specify the desired Large Language Model in --llm (e.g., --llm gpt-3.5-turbo). After running this file, the predictions are gonna be stored in the Results folder.


3) Run the "postprocess.py" file to extract the tactics' keywords from the LLM's response with the following line:
```python
python postprocess.py --file_path PATH
```
Specify the csv file path in --file_path created from the "main.py" file (e.g., --file_path ./results/preds_gpt-3.5-turbo_all_urls.csv). After running this file, the encoded predictions are gonna be stored in the Results folder.

4) For evalution, run "evaluation.py" file by specifying the encoded prediction file with the following line:
```python
python evaluation.py --encoded_file_path PATH
```
Specify the csv file path in --encoded_file_path created from the "postprocess.py" file (e.g., --encoded_file_path ./results/preds_gpt-3.5-turbo_all_urls_encoded.csv)


