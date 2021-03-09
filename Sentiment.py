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


all_articles = article.text

text = article.text.replace('\n', '')
SGG_sentences = text.split('.')



mobile_df = pd.DataFrame({'Sentences' : SGG_sentences, 'Model' : 'Samsung Galaxy Grand', 'Label' : 0} )

mobile_df['polarity'] = mobile_df.Sentences.apply(polarity_score)

mobile_df['subjectivity'] = mobile_df.Sentences.apply(subjectivity_score)

mobile_df['sent'] = mobile_df.Sentences.apply(sent_score)


#Nokia C7
nokia_c7_url = 'https://www.techradar.com/reviews/phones/mobile-phones/nokia-c7-905015/review'
article = Article(nokia_c7_url)
article.download()
article.parse()

all_articles += article.text

text = article.text.replace('\n', '')
NC7_sentences = text.split('.')

NC7_df = pd.DataFrame({'Sentences' : NC7_sentences, 'Model' : 'Nokia C7', 'Label' : 0} )

NC7_df = calculate_scores(NC7_df)


mobile_df = mobile_df.append(NC7_df)

#Asus Zenfone 5;
asus_zenfone5 = 'https://www.techradar.com/reviews/asus-zenfone-5-review'
article = Article(asus_zenfone5)
article.download()
article.parse()

all_articles += article.text

text = article.text.replace('\n', '')
AZ5_sentences = text.split('.')

AZ5_df = pd.DataFrame({'Sentences' : AZ5_sentences, 'Model' : 'Asus Zenphone 5', 'Label' : 1} )

AZ5_df = calculate_scores(AZ5_df)


mobile_df = mobile_df.append(AZ5_df)

#Zenfone 5;
motorola_moto_X = 'https://www.techradar.com/reviews/phones/mobile-phones/moto-x-1263345/review'
article = Article(motorola_moto_X)
article.download()
article.parse()

all_articles += article.text


text = article.text.replace('\n', '')
MX_sentences = text.split('.')

MX_df = pd.DataFrame({'Sentences' : MX_sentences, 'Model' : 'Motorola Moto X', 'Label' : 1} )

MX_df = calculate_scores(MX_df)


mobile_df = mobile_df.append(MX_df)

mobile_df[mobile_df['Model'] ==  'Asus Zenphone 5']['polarity'].mean()

mobile_df.groupby('Model').mean()

#Motorola MX Play
op_URL = 'https://www.techradar.com/sg/reviews/phones/mobile-phones/moto-x-play-1300372/review'
article = Article(op_URL)
article.download()
article.parse()


all_articles += article.text

text = article.text.replace('\n', '')
one_sentences = text.split('.')

mxp_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'Motorola Moto X Play', 'Label' : 2} )

mxp_df = calculate_scores(mxp_df)

mobile_df = mobile_df.append(mxp_df)

sam_URL = 'https://www.techradar.com/sg/reviews/samsung-galaxy-a8-review'
article2 = Article(sam_URL)
article2.download()
article2.parse()

all_articles += article2.text


text = article2.text.replace('\n', '')
sam_sentences = text.split('.')

sam_df = pd.DataFrame({'Sentences' : sam_sentences, 'Model' : 'Samsung Galaxy A8', 'Label' : 2} )

sam_df = calculate_scores(sam_df)

mobile_df = mobile_df.append(sam_df)


#Huawei Nexus 6P
hua_URL = 'https://www.techradar.com/sg/reviews/phones/mobile-phones/nexus-6p-1305318/review'
article2 = Article(hua_URL)
article2.download()
article2.parse()

all_articles += article2.text


text = article2.text.replace('\n', '')
hua_sentences = text.split('.')

hua_df = pd.DataFrame({'Sentences' : hua_sentences, 'Model' : 'Huawei Nexus 6P 2', 'Label' : 3} )

hua_df = calculate_scores(hua_df)


mobile_df = mobile_df.append(hua_df)


#Honor 6 plus
hon_URL = 'https://www.techradar.com/sg/reviews/phones/mobile-phones/honor-6-plus-1279376/review'
article = Article(hon_URL)
article.download()
article.parse()


all_articles += article.text


text = article.text.replace('\n', '')
one_sentences = text.split('.')

hon_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'Honor 6 plus X', 'Label' : 3} )

hon_df = calculate_scores(hon_df)


mobile_df = mobile_df.append(hon_df)


#LG Optimus G Pro
one_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/lg-optimus-g-pro-1148448/review'
article = Article(one_URL)
article.download()
article.parse()

all_articles += article.text


text = article.text.replace('\n', '')
one_sentences = text.split('.')
lg_df = pd.DataFrame({'Sentences' : one_sentences, 'Model' : 'LG Optimus G Pro', 'Label' : 4} )
lg_df = calculate_scores(lg_df)
mobile_df = mobile_df.append(lg_df)


#Asus Zenfone 2

ZTE_URL = 'https://www.techradar.com/reviews/phones/mobile-phones/asus-zenfone-2-1279756/review'
article2 = Article(ZTE_URL)
article2.download()
article2.parse()

all_articles += article2.text


text = article2.text.replace('\n', '')
ZTE_sentences = text.split('.')

ZTE_df = pd.DataFrame({'Sentences' : ZTE_sentences, 'Model' : 'Asus Zenfone 2', 'Label' : 4} )

ZTE_df = calculate_scores(ZTE_df)


mobile_df = mobile_df.append(ZTE_df)

#Replace labels so it matches the latest cluster decision
di = {0:1, 1:3, 2:5, 3:2, 4:4}
mobile_df.replace({'Label' : di}, inplace = True)

price_rank_dict = {1:1, 4:2, 5:3, 3:4, 2:5}  
mobile_df['price_rank'] = mobile_df['Label'].map(price_rank_dict)

#Summary Stats of Tech Radar DF
tr_means = mobile_df.groupby('Model').mean().sort_values('price_rank')


mobile_df.Model.unique()
#Subjectivity & Polarity
sns.set_palette(sns.color_palette("Paired", n_colors = 4))
sns.scatterplot(x = 'polarity', y = 'subjectivity', data = mobile_df, hue = 'Label', palette = sns.color_palette("tab10", n_colors = 5))
plt.title('Subjectivity & Polarity')
plt.show()

#Afinn vs TextBlob
sns.scatterplot(x = 'polarity', y = 'sent', data = mobile_df, hue = 'Label', palette = sns.color_palette("tab10", n_colors = 5))
plt.xlabel('Textblob Polarity')
plt.ylabel('Afinn Score')
plt.title('Afinn Score & Textblob Polarity')
plt.show()

#Wordcloud for all articles
wordcloud = WordCloud(max_font_size = 150, max_words=100, background_color="white",width=1000, height=500 ).generate(all_articles)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Polarity plot
g = sns.barplot(x = 'Label', y = 'polarity', data = tr_means,
palette ="tab10", order = [1,4,5,3,2])
plt.ylabel('Polarity')
plt.xlabel('Label (Ordered by price low - high)')
plt.title('Polarity Scores')
plt.show()

#Sentiment Score plot
g = sns.barplot(x = 'Label', y = 'sent', data = tr_means,
palette ="tab10", order = [1,4,5,3,2])
plt.ylabel('Sentiment')
plt.xlabel('Label (Ordered by price low - high)')
plt.title('Sentiment Scores')
plt.show()


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

#Word lengths of each review
def get_word_len(text):
    return len(text.split())

amazon_df['Word Count'] = amazon_df['Reviews'].apply(get_word_len)

amazon_df['polarity'] = amazon_df.Reviews.apply(polarity_score)
amazon_df['subjectivity'] = amazon_df.Reviews.apply(subjectivity_score)
amazon_df['sent'] = amazon_df.Reviews.apply(sent_score)

amazon_df.replace({'Label' : di}, inplace = True)


amzn_means_df = amazon_df.groupby('Model').mean()
amzn_means_df.sort_values('price_rank', inplace = True)





#Amazon User Rating plot
sns.barplot(data = amzn_means_df, x = 'Label', y = 'Rating' , ci=None,
palette ="tab10", order = [1,4,5,3,2])
plt.ylabel('User Rating')
plt.xlabel('Label (Ordered by price low - high)')
plt.title('User Reviews')
plt.show()

#Textblob
sns.barplot(data = amzn_means_df, x = 'Label', y = 'polarity' , ci=None,
palette ="tab10", order = [1,4,5,3,2])
plt.ylabel('Polarity')
plt.xlabel('Label (Ordered by price low - high)')
plt.title('TextBlob Rating')
plt.show()


#Afinn not included due to high variance
# sns.barplot(data = amzn_means_df, x = 'Label', y = 'sent' ,
# palette ="tab10", order = [1,4,5,3,2])
# plt.ylabel('Sentiment')
# plt.xlabel('Label (Ordered by price low - high)')
# plt.title('Afinn Rating')
# plt.show()


#Amazon Rating plot
sns.barplot(data = amzn_means_df, hue = 'Label', y = 'polarity', x = 'sent' )
# plt.show()

#Amazon subjectivity
sns.barplot(data = amzn_means_df, x = 'Label', y = 'subjectivity' , ci=None,
palette ="tab10", order = [1,4,5,3,2])
plt.ylabel('Subjectivity')
plt.xlabel('Label (Ordered by price low - high)')
plt.title('User Subjectivity Rating')
plt.ylim([0,1])
plt.show()

#TechRadar subjectivity
sns.barplot(data = tr_means, x = 'Label', y = 'subjectivity' , ci=None,
palette ="tab10", order = [1,4,5,3,2])
plt.ylabel('Subjectivity')
plt.xlabel('Label (Ordered by price low - high)')
plt.title('Expert Subjectivity Rating')
plt.ylim([0,1])
plt.show()

amzn_means_df.plot.bar(rot = 0)
plt.show()



amzn_means_df.drop(['Price','Label'], axis = 1).plot.bar(rot = 0)
plt.show()

word_below_200 = amazon_df[amazon_df['Word Count'] < 140]

corr = word_below_200.corr()
amazon_df.corr()


# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax =.5, vmin = -.5 )
plt.title('Correlation Matrix')
plt.show()


#Wordcountplot
sns.scatterplot(data = word_below_200, y = 'Word Count', x = 'polarity' )
plt.xlabel('Polarity')
plt.title('Word Count and Sentiment')
plt.show()

amazon_df[amazon_df['Word Count'] > 1000]

amazon_df['Reviews'][310432]