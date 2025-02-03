 # AI Recruitment Pipeline

This repository provides an AI-driven approach to analyze resumes, job descriptions, and transcripts for recruitment purposes. The pipeline leverages advanced text analysis techniques, including similarity measures, sentiment analysis, and other metrics, to aid in data-driven hiring decisions.

---

## Features

### **1. Data Loading**
- Loads input data from an Excel file (`data/prediction_data.xlsx`) with fields such as:
  - **Resume**
  - **Job Description**
  - **Transcript**
  - **Reason for Decision**
  - **Role**

### **2. Feature Extraction**
- **Resume and Job Description Similarity:** Calculates cosine similarity between the candidate's resume and the job description.
- **Resume and Transcript Similarity:** Computes cosine similarity between the resume and the interview transcript.
- **Role and Transcript Similarity:** Analyzes the alignment of the role description with the interview transcript.

### **3. Text Analysis**
- **Sentiment Analysis:** Performs sentiment analysis on transcripts using the VADER Sentiment Analyzer.
- **Lexical Diversity:** Calculates the diversity of vocabulary used in the transcript.
- **Text Length Metrics:** Extracts word counts for transcripts, resumes, and reasons for decisions.
- **Word Count Ratios:** Computes the ratio of transcript word count to resume word count.

### **4. Data Storage**
- Saves the processed data as a pickle file (`output/processed_prediction_data.pkl`) for further use.

### **5. Email Functionality**
- Sends an email with the processed data attached for easy sharing.

---

## Repository Structure

```plaintext
AI-Recruitment-Pipeline/
│
├── Jupyter Notebook/
│   ├── data_generation.ipynb      # Data preparation
│   ├── EDA.ipynb                  # Exploratory data analysis
│   ├── Training.ipynb             # Model training
│   ├── resume_screening.ipynb     # Screening logic
│   └── predict.ipynb              # Prediction pipeline
│
├── Python Codes/
│   ├── processed_prediction_data.pkl
│   ├── data_generation.py
│   ├── EDA.py
│   ├── Training.py
│   ├── Resume_screening.py
│   └── predict.py
│
├── data/
│   └── prediction_data.xlsx       # Input data file
│
├── output/
│   └── predicted_selection_status.xlsx  # Final predictions
│
├── README.md                      # Project documentation
└── LICENSE                        # License information

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Project Affiliation
This project was developed as part of the Infosys SpringBoard internship program.


