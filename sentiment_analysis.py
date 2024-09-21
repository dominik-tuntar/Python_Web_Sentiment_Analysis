import os
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nrclex import NRCLex

UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

def get_most_recent_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file

def clean_uploads_folder(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    for f in files:
        os.remove(f)

csv_file = get_most_recent_file(UPLOAD_FOLDER)
data = pd.read_csv(csv_file)
argument = 'content'
data = data.dropna(subset=[argument])
sia = SentimentIntensityAnalyzer()
plt.rc('font', size=6)

def analyze_sentiment(description):
    scores = sia.polarity_scores(description)
    return scores

data['Sentiment'] = data[argument].apply(analyze_sentiment)
data['Compound'] = data['Sentiment'].apply(lambda d: d['compound'])
data['Sentiment Category'] = pd.cut(data['Compound'], bins=[-1, -0.75, -0.2, 0.2, 0.75, 1], labels=['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive'])

sentiment_counts = data['Sentiment Category'].value_counts()

colors = {
    'Very negative': 'maroon',
    'Negative': 'red',
    'Neutral': 'beige',
    'Positive': 'lime',
    'Very positive': 'green'
}
fig, axes = plt.subplots(3, 2, figsize=(15, 8))
for category, color in colors.items():
    subset = data[data['Sentiment Category'] == category]
    subset['Compound'].plot(kind='hist', bins=20, color=color, alpha=0.5, label=category, ax=axes[0, 0])

axes[0, 0].set_title('Sentiment distribution')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Amount')
axes[0, 0].legend()

axes[0, 1].pie(sentiment_counts, labels=sentiment_counts.index, colors=[colors[label] for label in sentiment_counts.index], autopct='%1.1f%%', startangle=140)
axes[0, 1].set_title('Sentiment distribution (pie chart)')

def sentcateval(mean):
    category = ''
    if mean >= -1 and mean <= - 0.75 or mean < -1:
        category = '(Very negative)'
    elif mean >= -0.749999999 and mean <= -0.2:
        category = '(Negative)'
    elif mean >= -0.199999999 and mean <= 0.2:
        category = '(Neutral)'
    elif mean >= 0.200000001 and mean <= 0.75:
        category = '(Positive)'
    elif mean >= 0.7500000001 and mean <= 1 or mean > 1:
        category = '(Very positive)'
    return category

text_content = (
    'Median sentiment: ' + str(round(data['Compound'].mean(),3)) + ' ' + sentcateval(data['Compound'].mean()) + '\n' +
    '68% of sentiments are between ' + str(round(data['Compound'].mean() - data['Compound'].std(),3)) + ' ' + sentcateval(data['Compound'].mean() - data['Compound'].std()) + 
    ' and ' + str(round(data['Compound'].mean() + data['Compound'].std(),3)) + ' ' + sentcateval(data['Compound'].mean() + data['Compound'].std()) + '\n' +
    'Highest sentiment: ' + str(data['Compound'].max()) + '\n' +
    'Lowest sentiment: ' + str(data['Compound'].min())
)

axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5, text_content, ha='center', va='center', fontsize=12, wrap=True)

data[argument] = data[argument].fillna('').astype(str)
all_descriptions = ' '.join(data[argument].tolist())
tokens = word_tokenize(all_descriptions)
tokens = [word.lower() for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]
tags = nltk.pos_tag(tokens)
nouns = [word for word, pos in tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
freq_dist = FreqDist(nouns)
most_common_nouns = freq_dist.most_common(10)

words, counts = zip(*most_common_nouns)

axes[1, 1].bar(words, counts, color='blue')
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Most common words')
axes[1, 1].tick_params(axis='x', rotation=45)

text_object = NRCLex(all_descriptions)
emotion_scores = text_object.raw_emotion_scores

total_emotions = sum(emotion_scores.values())
emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_scores.items()}

custom_labels = {
    'fear': 'Fear',
    'anger': 'Anger',
    'anticipation': 'Anticipation',
    'trust': 'Trust',
    'surprise': 'Surprise',
    'positive': 'Positive',
    'negative': 'Negative',
    'sadness': 'Sadness',
    'disgust': 'Disgust',
    'joy': 'Joy'
}

custom_emotion_percentages = {custom_labels[emotion]: value for emotion, value in emotion_percentages.items()}

emotion_labels = list(custom_emotion_percentages.keys())
emotion_values = list(custom_emotion_percentages.values())

axes[2, 0].bar(emotion_labels, emotion_values, color=['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'grey'])
axes[2, 0].set_xlabel('Emotions')
axes[2, 0].set_ylabel('Percentage')
axes[2, 0].set_title('Distribution of emotions')

axes[2, 1].pie(emotion_values, labels=emotion_labels, autopct='%1.1f%%', colors=['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'grey'])
axes[2, 1].set_title('Distribution of emotions (pie chart)')

# Save the plot as an image file
plot_file_path = os.path.join(PLOT_FOLDER, 'sentiment_analysis_plot.png')
plt.tight_layout()
plt.savefig(plot_file_path)

# Clean up the plots and close the figure
plt.close(fig)

clean_uploads_folder(UPLOAD_FOLDER)

# Print the path to the saved plot file
print(plot_file_path)
