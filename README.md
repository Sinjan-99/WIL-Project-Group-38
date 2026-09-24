# WIL Group 38 - Australian Casual Employee Work-Rights RAG Assistant

## Group Information

**Group ID:** 38

## Group Members
| Student ID | Name | GitHub ID |
|---|---|---|
| S4152164 | Aanvita Ajaiy Awte | @Aanvita04 | 
| S4186051 | Amir Munawar Wangde| @aamir1901| 
| S4206979 | Kripa Anna Sajan | @kripa-sajan | 
| S4233261 | Krishi Kamlesh Patel | @Krishi301 | 
| S4222725 | Neha Merin George | @sphinx048 | 
| S4205755 | Sinjan Chakrabarty | @Sinjan-99 | 

## Project Overview
This project aims to develop a RAG system that helps casual employees in Australia to quickly find accurate information about their work rights and entitlements. This system is built and evaluated following the Walert methodology.

## Current Progress
- Walert RAG reproduction completed to the current retrieval-evaluation stage
- Walert BM25 retrieval index generated
- Known and Inferred evaluations completed
- Initial project scope and stakeholder identified
- Built Trello Board
- Australian Fair Work knowledge base created and validated
- Dense retrieval prototype implemented using Sentence Transformers
- Top-k retrieval tested on in-domain and out-of-domain questions

## Repository Structure
  
```text
data/       # Knowledge Base
src/        # RAG implementation
evaluation/ # Evaluation data and results
frontend/   # User interface
docs/       # Project documentation
```

## How to Run
### 1. Clone the repository
Clone the repository and open the project in VS Code:
```bash
git clone https://github.com/Sinjan-99/WIL-Project-Group-38.git
```
### 2. Create a Virtual Environment
Open the VS Code terminal in the project folder and run:
```bash
python -m venv .venv
```
### 3. Install Dependencies
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
### 4. Install Ollama
Install Ollama, then download the Llama 3.2 model:
```bash
ollama pull llama3.2
```
### 5. Run the Application
From the project root directory, run:
```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Knowledge Base
Processed knowledge base: data/processed/passages.csv
Raw Fair Work source pages: data/raw/
To rebuild the database:
```bash
.\.venv\Scripts\python.exe src/build_database.py
```

## Testing
The RAG pipeline can be tested using:
```bash
.\.venv\Scripts\python.exe src/pipeline_test.py
```
  
