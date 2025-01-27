import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
import pickle

# Load dataset files from 'datasets' directory
directory = "datasets"
data = pd.concat(
    [
        pd.read_excel(os.path.join(directory, file)).rename(
            columns=lambda x: x.strip().replace(" ", "_").translate(str.maketrans("", "", r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~"""))
        )
        for file in os.listdir(directory)
        if file.endswith((".xlsx", ".xls"))
    ],
    ignore_index=True, 
)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform job descriptions and resumes, transform transcript
jd_vectors = vectorizer.fit_transform(data['Job_Description'])
resume_vectors = vectorizer.transform(data['Resume'])  # Transform for consistency
transcript_vectors = vectorizer.transform(data['Transcript'])  # Transform

# Save the vectorizer model for future use
with open('tfidf_vectorization.pld', 'wb') as file:
    pickle.dump(vectorizer, file)

# Calculate cosine similarity between resume and job description for each candidate
data['resume_job_similarity'] = [cosine_similarity(resume_vectors[i], jd_vectors[i])[0][0] for i in range(len(data))]

# Calculate cosine similarity between transcript and job description for each candidate
data['transcript_jd_similarity'] = [cosine_similarity(transcript_vectors[i], jd_vectors[i])[0][0] for i in range(len(data))]

# Calculate cosine similarity between transcript and resume for each candidate
data['transcript_resume_similarity'] = [cosine_similarity(transcript_vectors[i], resume_vectors[i])[0][0] for i in range(len(data))]

# Sentiment analysis on each transcript using VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
data['sentiment'] = data['Transcript'].apply(lambda transcript: sia.polarity_scores(transcript)['compound'])

# Sentiment analysis for cultural fit based on Reason_for_decision using TextBlob
data['cultural_fit_sentiment'] = data['Reason_for_decision'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Confidence score based on phrases like 'I think' and 'Maybe'
data['confidence_score'] = data['Transcript'].apply(lambda x: x.count('I think') + x.count('Maybe'))

# Job fit analysis based on keywords overlap between job description and transcript
def job_fit_analysis(job_desc, transcript):
    job_keywords = job_desc.split()
    transcript_keywords = transcript.split()
    common_keywords = set(job_keywords).intersection(transcript_keywords)
    return len(common_keywords) / len(job_keywords)

data['job_fit_score'] = data.apply(lambda row: job_fit_analysis(row['Job_Description'], row['Transcript']), axis=1)

# Soft skills sentiment analysis using TextBlob
data['soft_skills_sentiment'] = data['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Clarity score using Flesch Reading Ease
data['clarity_score'] = data['Transcript'].apply(lambda x: textstat.flesch_reading_ease(x))

# Create bins for resume-job similarity
bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
labels = ['0-0.1', '0.1-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
data['similarity_bin'] = pd.cut(data['resume_job_similarity'], bins=bins, labels=labels, include_lowest=True)

# Group by similarity bin, decision (select/reject), and role, and count selections
role_decision_counts = data.groupby(['similarity_bin', 'decision', 'Role']).size().unstack(fill_value=0)

# Get the similarity bin threshold for selected candidates
selected_counts = role_decision_counts.xs(key='select', level='decision', axis=0)
thresholds = selected_counts.idxmax(axis=1).reset_index()
thresholds.columns = ['Role', 'Threshold_Similarity_Bin']

# Visualize similarity distribution by decision
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting similarity distribution grouped by decision
grouped_data = data.groupby(['similarity_bin', 'decision']).size().unstack(fill_value=0)
sns.heatmap(grouped_data, annot=True, fmt='d', cmap='Blues')
plt.title('Similarity Distribution Grouped by Decision')
plt.ylabel('Similarity Bin')
plt.xlabel('Decision (Select/Reject)')
plt.show()

# Display data and thresholds for further analysis
print(data[['Resume', 'Job_Description', 'transcript_jd_similarity', 'resume_job_similarity', 'decision']])
print(thresholds)
