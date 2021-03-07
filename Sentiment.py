#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt


from wordcloud import WordCloud
import re

# text imports

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from afinn import Afinn

from newspaper import Article


ssg_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/samsung-galaxy-grand-1134387/review'
article = Article(URL)
article.download()
article.parse()



wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

# # Display the  plot:
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

text = article.text.replace('\n', '')
SGG_sentences = text.split('.')

def polarity_score(text):
    p = TextBlob(text).sentiment.polarity
    return p

def subjectivity_score(text):
    s = TextBlob(text).sentiment.subjectivity
    return s

def sent_score(text):
    return afinn.score(text)


#Calculate Scores and complie them in a df
def calculate_scores(df):
    df = df.copy(deep = True)
    
    df['polarity'] = df.Sentences.apply(polarity_score)

    df['subjectivity'] = df.Sentences.apply(subjectivity_score)

    df['sent'] = df.Sentences.apply(sent_score)
    return df

mobile_df = pd.DataFrame({'Sentences' : SGG_sentences, 'Model' : 'Samsung Galaxy Grand', 'Label' : '0'} )

mobile_df['polarity'] = mobile_df.Sentences.apply(polarity_score)

mobile_df['subjectivity'] = mobile_df.Sentences.apply(subjectivity_score)

mobile_df['sent'] = mobile_df.Sentences.apply(sent_score)

mobile_df.plot.scatter('polarity', 'subjectivity')
plt.show()

sns.lmplot('polarity', 'sent', data = mobile_df)
plt.show()

#Nokia C7
nokia_c7_url = 'https://www.techradar.com/reviews/phones/mobile-phones/nokia-c7-905015/review'
article = Article(nokia_c7_url)
article.download()
article.parse()

wordcloud = wc.generate(article.text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()


text = article.text.replace('\n', '')
NC7_sentences = text.split('.')

NC7_df = pd.DataFrame({'Sentences' : NC7_sentences, 'Model' : 'Nokia C7', 'Label' : '0'} )

NC7_df = calculate_scores(NC7_df)

NC7_df.plot.scatter('polarity', 'subjectivity')
plt.show()

sns.lmplot('polarity', 'sent', data = NC7_df)
plt.show()
mobile_df = mobile_df.append(NC7_df)

#Asus Zenfone 5;
asus_zenfone5 = 'https://www.techradar.com/reviews/asus-zenfone-5-review'
article = Article(asus_zenfone5)
article.download()
article.parse()

wordcloud = wc.generate(article.text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()


text = article.text.replace('\n', '')
AZ5_sentences = text.split('.')

AZ5_df = pd.DataFrame({'Sentences' : AZ5_sentences, 'Model' : 'Asus Zenphone 5', 'Label' : '1'} )

AZ5_df = calculate_scores(AZ5_df)

AZ5_df.plot.scatter('polarity', 'subjectivity')
plt.show()

sns.lmplot('polarity', 'sent', data = AZ5_df)
plt.show()

mobile_df = mobile_df.append(AZ5_df)

#Zenfone 5;
motorola_moto_X = 'https://www.techradar.com/reviews/phones/mobile-phones/moto-x-1263345/review'
article = Article(motorola_moto_X)
article.download()
article.parse()

wordcloud = wc.generate(article.text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


text = article.text.replace('\n', '')
MX_sentences = text.split('.')

MX_df = pd.DataFrame({'Sentences' : MX_sentences, 'Model' : 'Motorola Moto X', 'Label' : '1'} )

MX_df = calculate_scores(MX_df)

MX_df.plot.scatter('polarity', 'subjectivity')
plt.show()

sns.lmplot('polarity', 'sent', data = MX_df)
plt.show()

mobile_df = mobile_df.append(MX_df)

mobile_df[mobile_df['Model'] ==  'Asus Zenphone 5']['polarity'].mean()

mobile_df.groupby('Model').mean()

