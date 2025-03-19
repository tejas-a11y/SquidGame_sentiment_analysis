import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import nltk
import re

df = pd.read_csv('tweets_v8.csv')

print(df.head())

print(df.shape)

print(df.isnull().sum())

df = df.drop(columns='user_location', axis=1)

print(df.head())

df = df.drop(columns='user_description', axis=1)

print(df.isnull().sum())

stemmer = nltk.SnowballStemmer('english')

nltk.download('stopwords')

from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

def variable(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [ x for x in text.split(' ') if x not in stopword]
    text = ' '.join(text)
    text = [stemmer.stem(x) for x in text.split(' ')]
    text = ' '.join(text)
    return text 

df['text'] = df['text'].apply(variable)

text = ' '.join(a for a in df.text)

stopwords = set(STOPWORDS)

cloud = WordCloud(stopwords=stopwords).generate(text)
plt.figure(figsize=(15, 15))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

nltk.download('vader_lexicon')

sentiment = SentimentIntensityAnalyzer()

df['positive'] = [sentiment.polarity_scores(a)['pos'] for a in df['text']]
df['negative'] = [sentiment.polarity_scores(a)['neg'] for a in df['text']]
df['neutral'] = [sentiment.polarity_scores(a)['neu'] for a in df['text']]

df = df[['text', 'positive', 'negative', 'neutral']]

print(df.head())

m = sum(df['positive'])
n = sum(df['negative'])
o = sum(df['neutral'])

def score(a, b, c):
    if (a>b) and (a>c):
        print('positive')
    elif (b>a) and (b>c):
        print('negative')
    else:
        print('neutral')

score(m, n, o)

print('positive', m)
print('negative', n)
print('neutral', o)