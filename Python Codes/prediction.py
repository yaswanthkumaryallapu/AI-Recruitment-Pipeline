import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
from textblob import TextBlob
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


predict_df=pd.read_excel('prediction_data.xlsx')

#Feature Extraction
vectorizer = TfidfVectorizer()
resume_jd_similarity = []
for i in range(len(predict_df)):
    resume = predict_df['Resume'][i]
    jd = predict_df['Job Description'][i]
    similarity = cosine_similarity(vectorizer.fit_transform([resume, jd]))[0, 1]
    resume_jd_similarity.append(similarity)
predict_df['resume_jd_similarity'] = resume_jd_similarity

# Calculate resume and transcript similarity (Cosine Similarity)
resume_transcript_similarity = []
for i in range(len(predict_df)):
    resume = predict_df['Resume'][i]
    transcript = predict_df['Transcript'][i]
    similarity = cosine_similarity(vectorizer.fit_transform([resume, transcript]))[0, 1]
    resume_transcript_similarity.append(similarity)
predict_df['resume_transcript_similarity'] = resume_transcript_similarity

# Perform sentiment analysis on each transcript
sia = SentimentIntensityAnalyzer()
predict_df['sentiment'] = predict_df['Transcript'].apply(lambda transcript: sia.polarity_scores(transcript)['compound'])

# Compute lexical diversity for each transcript
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)
predict_df['lexical_diversity'] = predict_df['Transcript'].apply(lexical_diversity)

# Length of transcript (number of words)
predict_df['transcript_length_words'] = predict_df['Transcript'].apply(lambda x: len(x.split()))

#Reason for Decision Length
predict_df['reason_length'] = predict_df['Reason for decision'].str.split().apply(len)

# Resume length (number of words)
predict_df['resume_length'] = predict_df['Resume'].apply(lambda x: len(x.split()))

#Word Count Ratio
predict_df['word_count_ratio'] = predict_df['transcript_length_words'] / predict_df['resume_length']

#Role to Transcript Similarity
def text_similarity(text1, text2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0, 0]
predict_df['role_transcript_similarity'] = predict_df.apply(
    lambda row: text_similarity(row['Role'], row['Transcript']), axis=1
)

#cultural_fit_sentiment
predict_df['cultural_fit_sentiment'] = predict_df['Reason for decision'].apply(lambda x: TextBlob(x).sentiment.polarity)

#Job Description to Transcript Sentiment Gap
predict_df['jd_transcript_sentiment_gap'] = predict_df['sentiment'] - predict_df['cultural_fit_sentiment']

#Job Description Length
predict_df['job_desc_length'] = predict_df['Job Description'].str.split().apply(len)

#Role to Resume Similarity
predict_df['role_resume_similarity'] = predict_df.apply(
    lambda row: text_similarity(row['Role'], row['Resume']), axis=1
)

#Combined Text Similarity
predict_df['combined_text_similarity'] = (
    predict_df['resume_jd_similarity'] + predict_df['resume_transcript_similarity']
) / 2

#Sentiment to Lexical Diversity Ratio
predict_df['sentiment_to_diversity_ratio'] = predict_df['sentiment'] / predict_df['lexical_diversity']

#clarity score
predict_df['clarity_score'] = predict_df['Transcript'].apply(lambda x: textstat.flesch_reading_ease(x))

#confidence score
predict_df['confidence_score'] = predict_df['Transcript'].apply(lambda x: x.count('I think') + x.count('Maybe'))

#Clarity and Confidence Interaction
predict_df['clarity_confidence_interaction'] = predict_df['clarity_score'] * predict_df['confidence_score']

#Soft Skills
predict_df['soft_skills_sentiment'] = predict_df['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Length of transcript (number of characters)
predict_df['transcript_length_characters'] = predict_df['Transcript'].apply(len)

# Calculate technical skill matching score
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
predict_df['technical_skill_match'] = predict_df.apply(lambda row: compute_similarity(row['Resume'], row['Job Description']), axis=1)

# Calculate technical skill matching score
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
predict_df['technical_skill_match2'] = predict_df.apply(lambda row: compute_similarity(row['Resume'], row['Transcript']), axis=1)

# Calculate technical skill matching score
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
predict_df['technical_skill_match3'] = predict_df.apply(lambda row: compute_similarity(row['Job Description'], row['Transcript']), axis=1)

# Job Description Experience Match (Simple matching based on keywords, could be improved)
predict_df['job_description_experience_match'] = predict_df.apply(lambda row: len(set(row['Resume'].split()) & set(row['Job Description'].split())), axis=1)

#job score
def job_fit_analysis(job_desc, transcript):
    # You can use similarity or keyword matching here
    job_keywords = job_desc.split()
    transcript_keywords = transcript.split()
    common_keywords = set(job_keywords).intersection(transcript_keywords)
    return len(common_keywords) / len(job_keywords)
predict_df['job_fit_score'] = predict_df.apply(lambda row: job_fit_analysis(row['Job Description'], row['Transcript']), axis=1)

#job description complexity
predict_df['job_desc_complexity'] = predict_df['Job Description'].apply(lambda x: textstat.flesch_reading_ease(x))

predict_df['num_words_in_transcript'] = predict_df['Transcript'].apply(lambda x: len(x.split()))

#interaction quality check
predict_df['interaction_quality'] = predict_df['num_words_in_transcript'] * predict_df['sentiment']

# Text complexity resume and transcript
def text_complexity(text):
    return len(text.split()) / len(set(text.split()))
predict_df['text_complexity_transcript'] = predict_df['Transcript'].apply(text_complexity)
predict_df['text_complexity_resume'] = predict_df['Resume'].apply(text_complexity)

# Create a fresh TF-IDF vectorizer and fit it to the data
Vectorizer = TfidfVectorizer()
Vectorizer.fit(predict_df['Job Description'].fillna(''))  # Fit on job descriptions

# Transform text data
new_job_desc_vectors = Vectorizer.transform(predict_df['Job Description'].fillna(''))
new_resume_vectors = Vectorizer.transform(predict_df['Resume'].fillna(''))
new_transcript_vectors = Vectorizer.transform(predict_df['Transcript'].fillna(''))

# Cosine similarities
predict_df['resume_jod_similarity'] = [
    cosine_similarity(new_resume_vectors[i], new_job_desc_vectors[i])[0][0] for i in range(len(predict_df))
]
predict_df['transcript_jod_similarity'] = [
    cosine_similarity(new_transcript_vectors[i], new_job_desc_vectors[i])[0][0] for i in range(len(predict_df))
]



# Load the best model from the saved file
with open('xgboost_best_model.pkl', 'rb') as model_file:
    xgb_best = pickle.load(model_file)

print("XGBoost model loaded successfully.")

# List of features that match the model training features
features = [
    'num_words_in_transcript', 'resume_jd_similarity',
    'resume_transcript_similarity', 'sentiment', 'lexical_diversity',
    'transcript_length_words', 'reason_length', 'resume_length',
    'word_count_ratio', 'role_transcript_similarity',
    'cultural_fit_sentiment', 'jd_transcript_sentiment_gap',
    'job_desc_length', 'role_resume_similarity', 'combined_text_similarity',
    'sentiment_to_diversity_ratio', 'clarity_score', 'confidence_score',
    'clarity_confidence_interaction', 'soft_skills_sentiment',
    'transcript_length_characters', 'technical_skill_match',
    'technical_skill_match2', 'technical_skill_match3',
    'job_description_experience_match', 'job_fit_score',
    'job_desc_complexity', 'interaction_quality',
    'text_complexity_transcript', 'text_complexity_resume'
]

# Ensure you're selecting only the relevant columns
X = predict_df[features]

# Convert data to numeric if needed
X = X.apply(pd.to_numeric, errors='coerce')

# Make predictions using the best XGBoost model
predict_df['Selection_Status'] = xgb_best.predict(X)

# Map the result to 'select' or 'reject'
predict_df['Selection_Status'] = predict_df['Selection_Status'].apply(lambda x: 'select' if x == 1 else 'reject')

# Save only specific columns 
predict_df[['Name','decision','Selection_Status']].to_excel('predicted_selection_status.xlsx', index=False)
print("Selected columns have been saved to 'predicted_selection_status.xlsx'.")

# Email credentials and setup
sender_email = "yallapuyaswanthkumar66@gmail.com"  # Your email address
receiver_email = "tharaknanda7@gmail.com"  # Receiver's email address
subject = "Predicted Selection Status"
body = "Please find attached the predicted selection status Excel file."

# Create the MIME message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject

# Attach the email body text
msg.attach(MIMEText(body, 'plain'))

# Attach the Excel file
file_path = 'predicted_selection_status.xlsx'
with open(file_path, 'rb') as file:
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename={file_path}')
    msg.attach(part)

# Connect to the SMTP server (Gmail example)
try:
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()  # Start TLS for security
        server.login(sender_email, "mvao ctgh ikue awdw")  # Login to your email account
        server.sendmail(sender_email, receiver_email, msg.as_string())  # Send email
        print("Email with attachment has been sent successfully.")
except Exception as e:
    print(f"Error: {e}")














