import os
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly.graph_objs import Scatter, Layout
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import nltk
nltk.download()

# Directory containing the Excel files
directory = "datasets"

# List all files in the directory
files = os.listdir(directory)

# Print the list of files
print(files)

import pandas as pd
# Directory containing the Excel files
directory = "datasets"

# List to store data from each Excel file
data_frames = []

# Loop through all files in the directory
for file in os.listdir(directory):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path)  # Read each Excel file into a DataFrame
        
        # Clean the column names by removing leading/trailing spaces and standardizing them
        df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces
        df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
        df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)  # Remove any non-alphanumeric characters

        # Append the DataFrame to the list
        data_frames.append(df)

# Combine all DataFrames into one
combo = pd.concat(data_frames, ignore_index=True)

# Print the cleaned and combined DataFrame
print(combo)

print(combo.columns.tolist())

print("Data Information:")
combo.info()
# Display the first few rows of the data
combo.head()

# Check for missing values in each column
missing_data = combo.isnull().sum()
print("Missing Data:\n", missing_data)

# Visualize missing data with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(combo.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Check for duplicate rows
duplicates = combo.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# If needed, you can drop duplicates
combo = combo.drop_duplicates()


# Statistical summary of numeric columns
print("Statistical Summary:")
print(combo.describe())

# Distribution of numerical columns
combo.hist(bins=20, figsize=(10, 8))
plt.show()

# Check unique values in 'decision' column (if it's categorical)
print("Unique Values in 'Decision' Column:")
print(combo['decision'].value_counts())

# Bar plot for categorical features (e.g., decision column)
plt.figure(figsize=(8, 6))
sns.countplot(x='decision', data=combo)
plt.title("Distribution of Decision Column")
plt.show()

# Example: Create a feature for sentiment score from a "Transcript" (using a sentiment analysis model)
def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

combo['sentiment_score'] = combo['Transcript'].apply(get_sentiment_score)
print(combo.columns.tolist())

combo.head()

# Distribution of numerical features
combo.hist(bins=30, figsize=(15, 10))
plt.suptitle("Distribution of Numerical Variables")
plt.show()

# Sentiment vs Decision (for cultural fit analysis)
plt.figure(figsize=(8, 6))
sns.boxplot(x='decision', y='sentiment_score', data=combo)
plt.title("Sentiment Score vs Decision")
plt.show()

# Generate word cloud for 'Transcript' column
text = ' '.join(combo['Transcript'].dropna())  # Join all transcripts into one large string
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Transcripts")
plt.show()

# Plotting sentiment polarity distribution from the 'Transcript' column
plt.figure(figsize=(8, 6))
sns.histplot(combo['sentiment_score'], kde=True, color='blue')
plt.title("Sentiment Score Distribution")
plt.show()

# Value counts for categorical features
print("Unique Values in 'Role' Column:")
print(combo['Role'].value_counts())

print("Unique Values in 'decision' Column:")
print(combo['decision'].value_counts())

# Plotting the distribution of categorical columns
plt.figure(figsize=(10, 6))
sns.countplot(x='Role', data=combo)
plt.title("Role Distribution")
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of 'num_words_in_transcript'
plt.figure(figsize=(8, 6))
sns.histplot(combo['num_words_in_transcript'], kde=True)
plt.title("Distribution of Words in Transcript")
plt.show()

# Analyze 'num_words_in_transcript' with respect to 'decision'
plt.figure(figsize=(8, 6))
sns.boxplot(x='decision', y='num_words_in_transcript', data=combo)
plt.title("Words in Transcript vs Decision")
plt.show()

# Analyze 'num_words_in_transcript' with respect to 'Role'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Role', y='num_words_in_transcript', data=combo)
plt.title("Words in Transcript vs Role")
plt.xticks(rotation=45)
plt.show()

# Create a crosstab between 'Role' and 'decision'
cross_tab_role_decision = pd.crosstab(combo['Role'], combo['decision'])
print("Crosstab between 'Role' and 'decision':")
print(cross_tab_role_decision)

# Visualize the crosstab as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab_role_decision, annot=True, cmap='coolwarm')
plt.title("Crosstab of Role and Decision")
plt.show()

# Assuming you have a sentiment score for each transcript
plt.figure(figsize=(8, 6))
sns.boxplot(x='Role', y='sentiment_score', data=combo)
plt.title("Sentiment Distribution per Role")
plt.xticks(rotation=45)
plt.show()


# Add a new feature for transcript length (word count)
combo['num_words_in_transcript'] = combo['Transcript'].apply(lambda x: len(str(x).split()))

# Boxplot showing transcript length by decision
plt.figure(figsize=(8, 6))
sns.boxplot(x='decision', y='num_words_in_transcript', data=combo)
plt.title("Transcript Length vs Decision")
plt.show()

# Role Distribution
fig = px.histogram(combo, x="Role", color="decision", title="Role Distribution by Decision")
fig.show()

# Decision Distribution
fig = px.pie(combo, names="decision", title="Decision Distribution")
fig.show()

# Filter the DataFrame for rows where the decision is "reject"
rejected_transcripts = combo[combo['decision'] == 'reject']['Transcript'].dropna()

# Combine all transcripts into a single string
text = ' '.join(rejected_transcripts)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Transcripts with Decision 'Reject'")
plt.show()

# Filter the DataFrame for rows where the decision is "select"
rejected_transcripts = combo[combo['decision'] == 'select']['Transcript'].dropna()

# Combine all transcripts into a single string
text = ' '.join(rejected_transcripts)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Transcripts with Decision 'Reject'")
plt.show()

# Generate a word cloud for 'Transcript' using Plotly
fig = px.scatter(
    combo,
    x='ID',  # Use ID for better differentiation
    y='num_words_in_transcript',
    size='num_words_in_transcript',
    color='decision',
    hover_name='Name',
    title="Interactive Word Cloud Visualization"
)
fig.show() 

# Plot sentiment scores, assuming 'sentiment_score' is the correct column
fig = px.histogram(combo, x="sentiment_score", color="decision", title="Sentiment Distribution by Decision")
fig.show()

# Filter transcripts with the term "decision" or "reject"
filtered_transcripts = combo['Transcript'].dropna().str.contains(r'\b(decision|reject)\b', case=False)
relevant_transcripts = combo['Transcript'][filtered_transcripts]
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(relevant_transcripts)
print(vectorizer.get_feature_names_out())


# Filter transcripts with the term "decision" or "select"
filtered_transcripts = combo['Transcript'].dropna().str.contains(r'\b(decision|select)\b', case=False)
relevant_transcripts = combo['Transcript'][filtered_transcripts]
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(relevant_transcripts)
print(vectorizer.get_feature_names_out())

# Analyze sentiment
analyzer = SentimentIntensityAnalyzer()
combo['vader_sentiment'] = combo['Transcript'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
print(combo[['Transcript', 'vader_sentiment']])

# Check data types and inspect the first few rows
print(combo[['vader_sentiment', 'sentiment_score']].head())

# If the columns are numerical, you can calculate their correlation
correlation = combo['vader_sentiment'].corr(combo['sentiment_score'])
print(f"Correlation between vader_sentiment and sentiment_score: {correlation}")

# Calculate summary statistics
vader_summary = combo['vader_sentiment'].describe()
sentiment_summary = combo['sentiment_score'].describe()

print("\nSummary Statistics for vader_sentiment:")
print(vader_summary)

print("\nSummary Statistics for sentiment_score:")
print(sentiment_summary)

# Visualize the comparison between the two columns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=combo['vader_sentiment'], y=combo['sentiment_score'])
plt.title('Comparison of vader_sentiment and sentiment_score')
plt.xlabel('vader_sentiment')
plt.ylabel('sentiment_score')
plt.show()

# Optionally, create a side-by-side boxplot for further comparison
plt.figure(figsize=(8, 6))
sns.boxplot(data=[combo['vader_sentiment'], combo['sentiment_score']], 
            notch=True, 
            vert=False, 
            patch_artist=True, 
            showmeans=True)
plt.yticks([0, 1], ['vader_sentiment', 'sentiment_score'])
plt.title('Boxplot of vader_sentiment and sentiment_score')
plt.show()






























