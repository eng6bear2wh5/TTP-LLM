# TTP-LLM
Advancing TTP Analysis: Harnessing the Power of Large Language Models with Retrieval Augmented Generation

This repository offers comprehensive resources and tools for analyzing the MITRE ATT&CK framework, specifically focused on classifying ATT&CK tactics. It is designed to facilitate the training and evaluation of various Large Language Models (LLMs) by employing fine-tuning techniques for encoder-only models and Retrieval-Augmented Generation (RAG) for decoder-only models.

## Setup
Create a conda environment and install the libraries:
```python
conda create --name TTP-LLM python=3.10
conda activate TTP-LLM
pip install -r requirements.txt
```

## Repository Structure

### Data Folder

1. **MITRE_Tactic_and_Techniques_Descriptions.csv**
   - **Description**: Training dataset crawled from the MITRE ATT&CK framework.
   - **Purpose**: Used for training encoder-only models to understand and classify tactics.

2. **MITRE_Procedures.csv**
   - **Description**: Main dataset crawled from the MITRE ATT&CK framework.
   - **Purpose**: Contains the procedures' descriptions along with their corresponding tactic(s).

3. **MITRE_Procedures_encoded.csv**
   - **Description**: Encoded version of the dataset for evaluation.

4. **Procedures_RAG_Similarity.csv**
   - **Description**: Contains the top-3 most similar procedures along with their tactics.

### Encoder-Only Folder

- **Notebooks**
  - **Purpose**: Fine-tuning encoder-only models (e.g., RoBERTa and SecureBERT) using the "MITRE_Tactic_and_Techniques_Descriptions.csv" dataset and evaluating them on "MITRE_Procedures.csv".

### Decoder-Only Folder

- **Purpose**: Instructions on using decoder-only LLMs with or without Retrieval-Augmented Generation (RAG) techniques.


1. **prompt_only.py**
   - **Description**: Script for directly accessing LLM models with prompt engineering.

2. **RAG.py**
   - **Description**: Script containing three different RAG techniques:
     - **all_urls**: Loads all the Enterprise URLs from MITRE ATT&CK for retrieval.
     - **reference_url**: Loads the reference URL of each specific procedure description (found in the MITRE_Procedures.csv dataset).
     - **similar_procedure_urls**: Retrieves URLs corresponding to the top-3 'target' procedure descriptions that are most similar to the 'source' procedure specified in the query.

## Usage

### How to Run

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


