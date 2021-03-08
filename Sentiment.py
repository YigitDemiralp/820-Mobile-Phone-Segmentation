#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
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



mobile_df = pd.DataFrame({'Sentences' : SGG_sentences, 'Model' : 'Samsung Galaxy Grand', 'Label' : 0} )

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

NC7_df = pd.DataFrame({'Sentences' : NC7_sentences, 'Model' : 'Nokia C7', 'Label' : 0} )

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

AZ5_df = pd.DataFrame({'Sentences' : AZ5_sentences, 'Model' : 'Asus Zenphone 5', 'Label' : 1} )

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

MX_df = pd.DataFrame({'Sentences' : MX_sentences, 'Model' : 'Motorola Moto X', 'Label' : 1} )

MX_df = calculate_scores(MX_df)

MX_df.plot.scatter('polarity', 'subjectivity')
#plt.show()

sns.lmplot('polarity', 'sent', data = MX_df)
#plt.show()

mobile_df = mobile_df.append(MX_df)

mobile_df[mobile_df['Model'] ==  'Asus Zenphone 5']['polarity'].mean()

mobile_df.groupby('Model').mean()

#Motorola MX Play
op_URL = 'https://www.techradar.com/sg/reviews/phones/mobile-phones/moto-x-play-1300372/review'
article = Article(op_URL)
article.download()
article.parse()
wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

# # Display the  plot:
plt.imshow(wordcloud)
plt.axis("off")
# plt.show()

text = article.text.replace('\n', '')
one_sentences = text.split('.')

mxp_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'Motorola Moto X Play', 'Label' : 2} )

mxp_df = calculate_scores(mxp_df)

mxp_df.plot.scatter('polarity', 'subjectivity')
# plt.show()

sns.lmplot('polarity', 'sent', data = mxp_df)
# plt.show()
mobile_df = mobile_df.append(mxp_df)

sam_URL = 'https://www.techradar.com/sg/reviews/samsung-galaxy-a8-review'
article2 = Article(sam_URL)
article2.download()
article2.parse()
wcd = WordCloud(background_color="white")
wordcloud2 = wc.generate(article2.text)

# # Display the  plot:
plt.imshow(wordcloud2)
plt.axis("off")
# plt.show()

text = article2.text.replace('\n', '')
sam_sentences = text.split('.')

sam_df = pd.DataFrame({'Sentences' : sam_sentences, 'Model' : 'Samsung Galaxy A8', 'Label' : 2} )

sam_df = calculate_scores(sam_df)

sam_df.plot.scatter('polarity', 'subjectivity')
# plt.show()

sns.lmplot('polarity', 'sent', data = sam_df)
# plt.show()
mobile_df = mobile_df.append(sam_df)


#Huawei Nexus 6P

hua_URL = 'https://www.techradar.com/sg/reviews/phones/mobile-phones/nexus-6p-1305318/review'
article2 = Article(hua_URL)
article2.download()
article2.parse()
wcd = WordCloud(background_color="white")
wordcloud2 = wc.generate(article2.text)

# # Display the  plot:
plt.imshow(wordcloud2)
plt.axis("off")
# plt.show()

text = article2.text.replace('\n', '')
hua_sentences = text.split('.')

hua_df = pd.DataFrame({'Sentences' : hua_sentences, 'Model' : 'Huawei Nexus 6P 2', 'Label' : 3} )

hua_df = calculate_scores(hua_df)

hua_df.plot.scatter('polarity', 'subjectivity')
# plt.show()

sns.lmplot('polarity', 'sent', data = hua_df)
# plt.show()
mobile_df = mobile_df.append(hua_df)


#Honor 6 plus
hon_URL = 'https://www.techradar.com/sg/reviews/phones/mobile-phones/honor-6-plus-1279376/review'
article = Article(hon_URL)
article.download()
article.parse()
wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

# # Display the  plot:
plt.imshow(wordcloud)
plt.axis("off")
# plt.show()

text = article.text.replace('\n', '')
one_sentences = text.split('.')

hon_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'Honor 6 plus X', 'Label' : 3} )

hon_df = calculate_scores(hon_df)

hon_df.plot.scatter('polarity', 'subjectivity')
# plt.show()

sns.lmplot('polarity', 'sent', data = hon_df)
# plt.show()
mobile_df = mobile_df.append(hon_df)


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
# plt.show()

text = article.text.replace('\n', '')
one_sentences = text.split('.')

lg_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'LG Optimus G Pro', 'Label' : 4} )

lg_df = calculate_scores(lg_df)
lg_df.plot.scatter('polarity', 'subjectivity')
# plt.show()

sns.lmplot('polarity', 'sent', data = lg_df)
# plt.show()

mobile_df = mobile_df.append(lg_df)


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
# plt.show()

text = article2.text.replace('\n', '')
ZTE_sentences = text.split('.')

ZTE_df = pd.DataFrame({'Sentences' : ZTE_sentences, 'Model' : 'Asus Zenfone 2', 'Label' : 4} )

ZTE_df = calculate_scores(ZTE_df)

ZTE_df.plot.scatter('polarity', 'subjectivity')
# plt.show()

sns.lmplot('polarity', 'sent', data = ZTE_df)
# plt.show()
mobile_df = mobile_df.append(ZTE_df)

mobile_df.groupby('Model').mean()

mobile_df.Model.unique()



#Amazon Dataset
amazon_reviews = pd.read_csv('Datasets\Amazon_Unlocked_Mobile.csv')

find_ssg = amazon_reviews['Product Name'].str.contains('Galaxy Grand')
amazon_df = amazon_reviews[(find_ssg)  & (amazon_reviews['Brand Name'] != 'Samsung Korea')]
amazon_df['Model'] = 'Galaxy Grand'
amazon_df['Label'] = 0

find_nokia_c7 = amazon_reviews['Product Name'].str.contains('Nokia C7')
temp_df = amazon_reviews[find_nokia_c7].copy()
temp_df['Model'] = 'Nokia C7'
temp_df['Label'] = 0
amazon_df = amazon_df.append(temp_df)


find_Z5 = amazon_reviews['Product Name'].str.contains('Zenfone 5')
temp_df = amazon_reviews[find_Z5].copy()
temp_df['Model'] = 'Zenfone 5'
temp_df['Label'] = 1
amazon_df = amazon_df.append(temp_df)


find_MX = amazon_reviews['Product Name'].str.contains('Moto X')
temp_df = amazon_reviews[find_MX].copy()
temp_df = temp_df[~temp_df["Product Name"].str.contains('Play')]
temp_df['Model'] = 'Moto X'
temp_df['Label'] = 1
amazon_df = amazon_df.append(temp_df)


find_MXP = amazon_reviews['Product Name'].str.contains('Moto X Play')
temp_df = amazon_reviews[find_MX].copy()
temp_df['Model'] = 'Moto X Play'
temp_df['Label'] = 2
amazon_df = amazon_df.append(temp_df)

def find_amazon(string, Label):
    global amazon_df
    find = amazon_reviews['Product Name'].str.contains(string)
    temp_df = amazon_reviews[find].copy(deep = True)
    temp_df['Model'] = string
    temp_df['Label'] = Label
    return amazon_df.append(temp_df)

amazon_df = find_amazon('Galaxy A8', 2)
amazon_df = find_amazon('Honor 6 Plus', 3)
amazon_df = find_amazon('Huawei Nexus 6P', 3)
amazon_df = find_amazon('Zenfone 2', 4)




amazon_df['polarity'] = amazon_df.Reviews.apply(polarity_score)
amazon_df['subjectivity'] = amazon_df.Reviews.apply(subjectivity_score)
amazon_df['sent'] = amazon_df.Reviews.apply(sent_score)

means_df = amazon_df.groupby('Model').mean()
means_df.sort_values('Label', inplace = True)
means_df

#Amazon Rating plot
sns.barplot(data = means_df, x = 'Label', y = 'Rating' )
plt.show()

#Amazon Rating plot
sns.barplot(data = means_df, hue = 'Label', y = 'polarity', x = 'sent' )
# plt.show()





means_df.plot.bar(rot = 0)
plt.show()



means_df.drop(['Price','Label'], axis = 1).plot.bar(rot = 0)
plt.show()