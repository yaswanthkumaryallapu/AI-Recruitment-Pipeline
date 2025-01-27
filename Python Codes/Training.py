import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import PartialDependenceDisplay



directory = "datasets"
files = os.listdir(directory)

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

# Calculate resume and job description similarity (Cosine Similarity)
vectorizer = TfidfVectorizer()
resume_jd_similarity = []
for i in range(len(data)):
    resume = data['Resume'][i]
    jd = data['Job_Description'][i]
    similarity = cosine_similarity(vectorizer.fit_transform([resume, jd]))[0, 1]
    resume_jd_similarity.append(similarity)
data['resume_jd_similarity'] = resume_jd_similarity

# Calculate resume and transcript similarity (Cosine Similarity)
resume_transcript_similarity = []
for i in range(len(data)):
    resume = data['Resume'][i]
    transcript = data['Transcript'][i]
    similarity = cosine_similarity(vectorizer.fit_transform([resume, transcript]))[0, 1]
    resume_transcript_similarity.append(similarity)
data['resume_transcript_similarity'] = resume_transcript_similarity

# Perform sentiment analysis on each transcript
sia = SentimentIntensityAnalyzer()
data['sentiment'] = data['Transcript'].apply(lambda transcript: sia.polarity_scores(transcript)['compound'])
average_sentiment = data['sentiment'].mean()

# Function to calculate lexical diversity
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

# Compute lexical diversity for each transcript
data['lexical_diversity'] = data['Transcript'].apply(lexical_diversity)

# Length of transcript (number of words)
data['transcript_length_words'] = data['Transcript'].apply(lambda x: len(x.split()))

#Reason for Decision Length
data['reason_length'] = data['Reason_for_decision'].str.split().apply(len)

# Resume length (number of words)
data['resume_length'] = data['Resume'].apply(lambda x: len(x.split()))

#Word Count Ratio
data['word_count_ratio'] = data['transcript_length_words'] / data['resume_length']

#Role to Transcript Similarity
def text_similarity(text1, text2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0, 0]
data['role_transcript_similarity'] = data.apply(
    lambda row: text_similarity(row['Role'], row['Transcript']), axis=1
)

#culture fit sentiment
data['cultural_fit_sentiment'] = data['Reason_for_decision'].apply(lambda x: TextBlob(x).sentiment.polarity)

#Job Description to Transcript Sentiment Gap
data['jd_transcript_sentiment_gap'] = data['sentiment'] - data['cultural_fit_sentiment']

#Job Description Length
data['job_desc_length'] = data['Job_Description'].str.split().apply(len)

#Role to Resume Similarity
data['role_resume_similarity'] = data.apply(
    lambda row: text_similarity(row['Role'], row['Resume']), axis=1
)

#Combined Text Similarity
data['combined_text_similarity'] = (
    data['resume_jd_similarity'] + data['resume_transcript_similarity']
) / 2
data['sentiment_to_diversity_ratio'] = data['sentiment'] / data['lexical_diversity']

#Clarity Score
data['clarity_score'] = data['Transcript'].apply(lambda x: textstat.flesch_reading_ease(x))

#confidence score
data['confidence_score'] = data['Transcript'].apply(lambda x: x.count('I think') + x.count('Maybe'))

#Clarity and Confidence Interaction
data['clarity_confidence_interaction'] = data['clarity_score'] * data['confidence_score']

#Soft Skills Sentiment
data['soft_skills_sentiment'] = data['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Length of transcript (number of characters)
data['transcript_length_characters'] = data['Transcript'].apply(len)

# Function to compute similarity score between Resume and Job Description
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
data['technical_skill_match'] = data.apply(lambda row: compute_similarity(row['Resume'], row['Job_Description']), axis=1)

# Function to compute similarity score between Resume and Job Description
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
data['technical_skill_match2'] = data.apply(lambda row: compute_similarity(row['Resume'], row['Transcript']), axis=1)

# Function to compute similarity score between Resume and Job Description
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
data['technical_skill_match3'] = data.apply(lambda row: compute_similarity(row['Job_Description'], row['Transcript']), axis=1)

# Job Description Experience Match (Simple matching based on keywords, could be improved)
data['job_description_experience_match'] = data.apply(lambda row: len(set(row['Resume'].split()) & set(row['Job_Description'].split())), axis=1)

#job score
def job_fit_analysis(job_desc, transcript):
    # You can use similarity or keyword matching here
    job_keywords = job_desc.split()
    transcript_keywords = transcript.split()
    common_keywords = set(job_keywords).intersection(transcript_keywords)
    return len(common_keywords) / len(job_keywords)
data['job_fit_score'] = data.apply(lambda row: job_fit_analysis(row['Job_Description'], row['Transcript']), axis=1)

#Job Description Complexity
data['job_desc_complexity'] = data['Job_Description'].apply(lambda x: textstat.flesch_reading_ease(x))

#num_words_in_transcript
data['num_words_in_transcript'] = data['Transcript'].apply(lambda x: len(x.split()))

#interaction quality check
data['interaction_quality'] = data['num_words_in_transcript'] * data['sentiment']

# Text complexity (resume and transcript - using a simple metric like Flesch Reading Ease)
def text_complexity(text):
    return len(text.split()) / len(set(text.split()))  # A basic metric
data['text_complexity_transcript'] = data['Transcript'].apply(text_complexity)
data['text_complexity_resume'] = data['Resume'].apply(text_complexity)

# Encoding the target variable (select/reject)
le = LabelEncoder()
data['decision'] = le.fit_transform(data['decision'])  # 0: reject, 1: select


#Select features for model training (removed the removed features)
X = data[[
    'resume_jd_similarity',
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
       'job_desc_complexity', 'num_words_in_transcript', 'interaction_quality',
       'text_complexity_transcript', 'text_complexity_resume']
]
y = data['decision']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#xgboost model
param_grid = {
    'max_depth': [3, 5, 7],              
    'learning_rate': [0.01, 0.1, 0.3],   
    'n_estimators': [50, 100, 150]      
}
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
# Retrieve the best model and parameters
xgb_best = grid_search.best_estimator_
best_params = grid_search.best_params_
# Make predictions
xgb_y_pred = xgb_best.predict(X_test)
xgb_y_pred_prob = xgb_best.predict_proba(X_test)[:, 1]
# Evaluate model performance
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_roc_auc = roc_auc_score(y_test, xgb_y_pred_prob)
# Display the results
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"XGBoost ROC AUC: {xgb_roc_auc:.4f}")


# Error Analysis
xgb_errors = X_test.copy()
xgb_errors['True Label'] = y_test
xgb_errors['Predicted Label'] = xgb_best.predict(X_test)
xgb_errors['Error'] = xgb_errors['True Label'] != xgb_errors['Predicted Label']
misclassified_xgb = xgb_errors[xgb_errors['Error']]
print(f"Misclassified Instances:{len(misclassified_xgb)}")

# Impact Analysis (using feature importances)
xgb_feature_importance = xgb_best.get_booster().get_score(importance_type='weight')
xgb_impact_analysis = pd.DataFrame(
    list(xgb_feature_importance.items()), columns=['Feature', 'Importance']
).sort_values(by='Importance', ascending=False)
print("Impact Analysis (XGBoost):",xgb_impact_analysis)

# --- XGBoost Feature Importance Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_impact_analysis)
plt.title('XGBoost Feature Importance')
plt.show()

# --- Summary ---
top_xgb_features = xgb_impact_analysis.head(5) 
print("Top 5 most impactful features:")
for i, row in top_xgb_features.iterrows():
    print(f"{row['Feature']}: Importance = {row['Importance']:.4f}")

#shap values
# Ensure you're using the best model from Gradient Boosting Classifier (gb_best)
explainer = shap.Explainer(xgb_best, X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train)

# Beeswarm plot for SHAP values
shap.plots.beeswarm(shap_values)

# Waterfall plot for the first instance
base_value  = explainer.expected_value
print(f"Base Value: {base_value}")


shap_values_test = explainer(X_test)  

# Iterate through the first 3 instances
for i in range(3):
    instance_index = i
    shap_value = shap_values_test[instance_index]
    
    # Generate the waterfall plot
    print(f"\n--- SHAP Waterfall Plot for Instance {instance_index + 1} ---")
    shap.plots.waterfall(shap_value)
    
    # Extract information for summary
    feature_contributions = shap_value.values
    base_value = shap_value.base_values 
    predicted_value = base_value + feature_contributions.sum() 
    feature_names = shap_value.feature_names
    top_features = sorted(zip(feature_names, feature_contributions), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Convert log-odds to probabilities
    base_probability = 1 / (1 + np.exp(-base_value))  
    predicted_probability = 1 / (1 + np.exp(-predicted_value)) 
    
    # Print summary
    print("\nSummary:")
    print(f"Base Probability: {base_probability:.4%}")
    print(f"Predicted Probability: {predicted_probability:.4%}")
    
    print("\nTop Contributing Features:")
    for feature, contribution in top_features:
        direction = "increases" if contribution > 0 else "decreases"
        print(f"  - {feature}: {contribution:.4f} ({direction} prediction)")
    
    #Feature value insight
    print("\nFeature Value Insights:")
    for feature, contribution in top_features:
        feature_index = feature_names.index(feature)
        feature_value = shap_value.data[feature_index]
        print(f"  - {feature} has a value of {feature_value:.4f}, contributing {contribution:.4f}.")


# Dependency Plot Feature 1
features_to_plot = "resume_transcript_similarity" 
shap.dependence_plot(
    ind=features_to_plot, 
    shap_values=shap_values_test.values,  
    features=X_test.values,  
    feature_names=X_test.columns, 
    interaction_index=None, 
    cmap=plt.cm.Reds  
)

# Dependency Plot Feature 2
feature_name = "clarity_score"  

shap.dependence_plot(
    ind=feature_name, 
    shap_values=shap_values_test.values,  
    features=X_test.values,  
    feature_names=X_test.columns, 
    interaction_index=None, 
    cmap=plt.cm.Reds  
)

# Dependency Plot Feature 3
feature_name = "job_desc_length"  

shap.dependence_plot(
    ind=feature_name, 
    shap_values=shap_values_test.values,  
    features=X_test.values,  
    feature_names=X_test.columns, 
    interaction_index=None, 
    cmap=plt.cm.Reds  
)

#PartialDependence Plot for Feature
feature_name = "text_complexity_transcript"
feature_index = X_test.columns.get_loc(feature_name)

PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_test, 
    features=[feature_index], 
    feature_names=X_test.columns,  
    grid_resolution=50,  
    kind="average" 
)
plt.title(f"Partial Dependence Plot for {feature_name}")
plt.show()

#PartialDependence Plot for Feature2
feature_name = "soft_skills_sentiment"
feature_index = X_test.columns.get_loc(feature_name)

PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_test,  
    features=[feature_index], 
    feature_names=X_test.columns,  
    grid_resolution=50,
    kind="average"  
)
plt.title(f"Partial Dependence Plot for {feature_name}")
plt.show()

#PartialDependence Plot for Feature3
feature_name = "job_description_experience_match"
feature_index = X_test.columns.get_loc(feature_name)

PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_test, 
    features=[feature_index], 
    feature_names=X_test.columns, 
    grid_resolution=50,  
    kind="average"  # Fixed the typo here
)
plt.title(f"Partial Dependence Plot for {feature_name}")
plt.show()

#2D Partial Dependence Plot
features = [('resume_jd_similarity', 'resume_transcript_similarity')]  # 2D feature tuple
print("\n--- 2D Partial Dependence Plot for 'resume_jd_similarity' and 'resume_transcript_similarity' ---")
PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_train,  
    features=features, 
    grid_resolution=50,
    kind='average',
)
plt.suptitle("2D Partial Dependence Plot", fontsize=16)
plt.tight_layout()
plt.show()

#2D Partial Dependence Plot for Feature2
features = [('job_fit_score', 'sentiment')]  # 2D feature tuple
print("\n--- 2D Partial Dependence Plot for 'resume_jd_similarity' and 'resume_transcript_similarity' ---")
PartialDependenceDisplay.from_estimator(
    estimator=xgb_best, 
    X=X_train, 
    features=features,  
    grid_resolution=50, 
    kind='average', 
)
plt.suptitle("2D Partial Dependence Plot", fontsize=16)
plt.tight_layout()
plt.show()


