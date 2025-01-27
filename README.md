AI Recruitment Pipeline
This repository provides an AI-driven approach to analyze resumes, job descriptions, and transcripts for recruitment purposes. The pipeline leverages advanced text analysis techniques, including similarity measures, sentiment analysis, and other metrics, to aid in data-driven hiring decisions.

Features
1. Data Loading
Loads input data from an Excel file (data/prediction_data.xlsx) with fields such as:
Resume
Job Description
Transcript
Reason for Decision
Role
2. Feature Extraction
Resume and Job Description Similarity: Calculates cosine similarity between the candidate's resume and the job description.
Resume and Transcript Similarity: Computes cosine similarity between the resume and the interview transcript.
Role and Transcript Similarity: Analyzes the alignment of the role description with the interview transcript.
3. Text Analysis
Sentiment Analysis: Performs sentiment analysis on transcripts using the VADER Sentiment Analyzer.
Lexical Diversity: Calculates the diversity of vocabulary used in the transcript.
Text Length Metrics: Extracts word counts for transcripts, resumes, and reasons for decisions.
Word Count Ratios: Computes the ratio of transcript word count to resume word count.
4. Data Storage
Processed data is saved as a pickle file (output/processed_prediction_data.pkl) for further use.
5. Email Functionality
Sends an email with the processed data attached, making it easier to share insights.
Repository Structure
bash
Copy
Edit
AI-Recruitment-Pipeline/
│
├── Jupyter Notebook/
│   └── data_generation.ipynb
│   └── EDA.ipynb
│   └── Training.ipynb
│   └── resume_screeing.ipynb
│   └── predict.ipynb
│
├── Python Codes/
│   └── processed_prediction_data.pkl
│   └──data_generation.py
│   └── EDA.py
│   └── Training.py
│   └── Resume_screeing.py
│   └── predict.py
│
├── prediciton_data.xlsx
├──predicted_selection_status.xlsx
│                  
│              
├── README.md                     

