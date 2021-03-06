# -*- coding: utf-8 -*-
"""sentiment analysis LG&Asus.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1paXjYd7emOkG-ENIU5L_G-qMiDV8cNL3
"""

!pip install scikit-plot
!pip install afinn
!pip install newspaper3k

#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot


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
from textblob import TextBlob

#LG Optimus G Pro
one_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/lg-optimus-g-pro-1148448/review'
article = Article(one_URL)
article.download()
article.parse()
wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

# # Display the  plot:
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

text = article.text.replace('\n', '')
one_sentences = text.split('.')

def polarity_score(text):
    p = TextBlob(text).sentiment.polarity
    return p

def subjectivity_score(text):
    s = TextBlob(text).sentiment.subjectivity
    return s

def sent_score(text):
    afinn=Afinn()
    return afinn.score(text)

#Calculate Scores and complie them in a df
def calculate_scores(df):
    df = df.copy(deep = True)
    
    df['polarity'] = df.Sentences.apply(polarity_score)

    df['subjectivity'] = df.Sentences.apply(subjectivity_score)

    df['sent'] = df.Sentences.apply(sent_score)
    return df

mobile_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'LG Optimus G Pro', 'Label' : '0'} )

mobile_df['polarity'] = mobile_df.Sentences.apply(polarity_score)

mobile_df['subjectivity'] = mobile_df.Sentences.apply(subjectivity_score)

mobile_df['sent'] = mobile_df.Sentences.apply(sent_score)

mobile_df.plot.scatter('polarity', 'subjectivity')
plt.show()

sns.lmplot('polarity', 'sent', data = mobile_df)
plt.show()

#Asus Zenfone 2

ZTE_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/asus-zenfone-2-1279756/review'
article2 = Article(ZTE_URL)
article2.download()
article2.parse()
wcd = WordCloud(background_color="white")
wordcloud2 = wc.generate(article2.text)

# # Display the  plot:
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

text = article2.text.replace('\n', '')
ZTE_sentences = text.split('.')

ZTE_df = pd.DataFrame({'Sentences' : ZTE_sentences, 'Model' : 'Asus Zenfone 2', 'Label' : '0'} )

ZTE_df = calculate_scores(ZTE_df)

ZTE_df.plot.scatter('polarity', 'subjectivity')
plt.show()

sns.lmplot('polarity', 'sent', data = ZTE_df)
plt.show()
mobile_df = mobile_df.append(ZTE_df)

mobile_df.groupby('Model').mean()











