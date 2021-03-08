#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from textblob import TextBlob

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

afinn = Afinn(language='en')
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

ssg_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/samsung-galaxy-grand-1134387/review'
article = Article(ssg_URL)
article.download()
article.parse()



wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

# # Display the  plot:
plt.imshow(wordcloud)
plt.axis("off")
##plt.show()

text = article.text.replace('\n', '')
SGG_sentences = text.split('.')



mobile_df = pd.DataFrame({'Sentences' : SGG_sentences, 'Model' : 'Samsung Galaxy Grand', 'Label' : '0'} )

mobile_df['polarity'] = mobile_df.Sentences.apply(polarity_score)

mobile_df['subjectivity'] = mobile_df.Sentences.apply(subjectivity_score)

mobile_df['sent'] = mobile_df.Sentences.apply(sent_score)

mobile_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = mobile_df)
#plt.show()

#Nokia C7
nokia_c7_url = 'https://www.techradar.com/reviews/phones/mobile-phones/nokia-c7-905015/review'
article = Article(nokia_c7_url)
article.download()
article.parse()

wordcloud = wc.generate(article.text)

plt.imshow(wordcloud)
plt.axis("off")
#plt.show()


text = article.text.replace('\n', '')
NC7_sentences = text.split('.')

NC7_df = pd.DataFrame({'Sentences' : NC7_sentences, 'Model' : 'Nokia C7', 'Label' : '0'} )

NC7_df = calculate_scores(NC7_df)

NC7_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = NC7_df)
#plt.show()
mobile_df = mobile_df.append(NC7_df)

#Asus Zenfone 5;
asus_zenfone5 = 'https://www.techradar.com/reviews/asus-zenfone-5-review'
article = Article(asus_zenfone5)
article.download()
article.parse()

wordcloud = wc.generate(article.text)

plt.imshow(wordcloud)
plt.axis("off")
#plt.show()


text = article.text.replace('\n', '')
AZ5_sentences = text.split('.')

AZ5_df = pd.DataFrame({'Sentences' : AZ5_sentences, 'Model' : 'Asus Zenphone 5', 'Label' : '1'} )

AZ5_df = calculate_scores(AZ5_df)

AZ5_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = AZ5_df)
#plt.show()

mobile_df = mobile_df.append(AZ5_df)

#Zenfone 5;
motorola_moto_X = 'https://www.techradar.com/reviews/phones/mobile-phones/moto-x-1263345/review'
article = Article(motorola_moto_X)
article.download()
article.parse()

wordcloud = wc.generate(article.text)
plt.imshow(wordcloud)
plt.axis("off")
#plt.show()


text = article.text.replace('\n', '')
MX_sentences = text.split('.')

MX_df = pd.DataFrame({'Sentences' : MX_sentences, 'Model' : 'Motorola Moto X', 'Label' : '1'} )

MX_df = calculate_scores(MX_df)

MX_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = MX_df)
#plt.show()

mobile_df = mobile_df.append(MX_df)

mobile_df[mobile_df['Model'] ==  'Asus Zenphone 5']['polarity'].mean()

mobile_df.groupby('Model').mean()


one_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/oneplus-x-1307733/review'
article = Article(one_URL)
article.download()
article.parse()

wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

plt.imshow(wordcloud)
plt.axis("off")
#plt.show()

text = article.text.replace('\n', '')
one_sentences = text.split('.')


oneplus_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'One Plus X', 'Label' : '4'} )

oneplus_df['polarity'] = oneplus_df.Sentences.apply(polarity_score)

oneplus_df['subjectivity'] = oneplus_df.Sentences.apply(subjectivity_score)

oneplus_df['sent'] = oneplus_df.Sentences.apply(sent_score)

oneplus_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = oneplus_df)
#plt.show()

mobile_df = mobile_df.append(oneplus_df)

ZTE_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/zte-star-2-1286096/review'
article2 = Article(ZTE_URL)
article2.download()
article2.parse()
wcd = WordCloud(background_color="white")
wordcloud2 = wc.generate(article2.text)

plt.imshow(wordcloud2)
plt.axis("off")
#plt.show()


text = article2.text.replace('\n', '')
ZTE_sentences = text.split('.')

ZTE_df = pd.DataFrame({'Sentences' : ZTE_sentences, 'Model' : 'ZTE Star 2', 'Label' : '4'} )

ZTE_df = calculate_scores(ZTE_df)

ZTE_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = ZTE_df)
#plt.show()
mobile_df = mobile_df.append(ZTE_df)

mobile_df.Model.unique()

amazon_reviews = pd.read_csv('Datasets\Amazon_Unlocked_Mobile.csv')

find_ssg = amazon_reviews['Product Name'].str.contains('Galaxy Grand')
amazon_df = amazon_reviews[(find_ssg)  & (amazon_reviews['Brand Name'] != 'Samsung Korea')]
amazon_df['Model'] = 'Galaxy Grand'

find_nokia_c7 = amazon_reviews['Product Name'].str.contains('Nokia C7')
temp_df = amazon_reviews[find_nokia_c7].copy()
temp_df['Model'] = 'Nokia C7'
amazon_df = amazon_df.append(temp_df)


find_Z5 = amazon_reviews['Product Name'].str.contains('Zenfone 5')
temp_df = amazon_reviews[find_MX].copy()
temp_df['Model'] = 'Zenfone 5'
amazon_df = amazon_df.append(temp_df)


find_MX = amazon_reviews['Product Name'].str.contains('Moto X')
temp_df = amazon_reviews[find_MX].copy()
temp_df['Model'] = 'Moto X'
amazon_df = amazon_df.append(temp_df)

amazon_df.groupby('Model').mean()


amazon_df['polarity'] = amazon_df.Reviews.apply(polarity_score)

amazon_df['subjectivity'] = amazon_df.Reviews.apply(subjectivity_score)

amazon_df['sent'] = amazon_df.Reviews.apply(sent_score)

