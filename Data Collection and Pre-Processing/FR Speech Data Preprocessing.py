#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:57:21 2023

@author: lancelotpan
"""

import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.common.by import By
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import datetime
import re
import spacy
 



def scrap_FR_speech():
    # scrap multiple FR speech text data
    driver = webdriver.Chrome()
    
    # Navigate to a webpage with pagination links
    driver.get("https://www.federalreserve.gov/newsevents/speeches.htm")
    
    url=[]
    key_words = []
    FR_speech = pd.DataFrame()
    
    for page in range(1,24):
        if page == 24:
            passage_num = 13
        else:
            passage_num = 21
        
        for x in range(1,passage_num):
            url= driver.find_element(By.XPATH,'//*[@id="article"]/div[1]/div['+str(x)+']/div[2]/p[1]/em/a').get_attribute('href')
            # article text & keywords
            article = Article(url)
            article.download()
            article.parse()
            t = article.text
            # article.nlp()
            # k = article.keywords
            df = pd.DataFrame([t], columns=['minutes'] )
            df['url'] = url
            # get article time
            response = requests.get(url)
            html_content = response.text
            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            article_time = soup.find('p',class_='article__time').text
            article_time = datetime.datetime.strptime(article_time,"%B %d, %Y")
            df['article_time'] = article_time
            FR_speech= FR_speech.append(df)
            # key_words.append(k)
        driver.find_element(By.LINK_TEXT,"Next").click()
    #  xpath: //*[@id="article"]/div[1]/div[1]/div[2]/p[1]/em/a
    #         //*[@id="article"]/div[1]/div[2]/div[2]/p[1]/em/a
    return FR_speech
        
FR_speech = scrap_FR_speech()
FR_speech.to_parquet('FR_speech.parquet')


# FR_speech = FR_speech.drop(columns=['index', 'minutes_cleaned','minutes_c', 'Sentences',
#        'old_length', 'new_length', 'new_sentences', 'processed_text',
#        'minutes_cleaned'])
# FR_speech.to_parquet('FR_speech.parquet')
# FR_speech.reset_index(drop=True)


# Data cleaning
FR_speech = pd.read_parquet('/Users/lancelotpan/Desktop/Module 4/NLP/Mfin 7036 group project/FR_speech.parquet')

FR_speech = FR_speech.reset_index(drop=True)

FR_speech['minutes_cleaned'] = pd.Series()

# Drop reference in speech text

for count, item in enumerate(FR_speech['minutes']):
   lst = item.split('\n')
   item = ''.join(lst)
   # drop reference
   lst = item.split('Reference')
   item = lst[:1]
   item = "".join(item)
   # drop notes
   lst = item.split('. Return to text')
   item = lst[:1]
   item = "".join(item)
   FR_speech['minutes_cleaned'][count] = item
   
   
# Drop useless characters and convert all words to lowercase

def pre_process(text):
    
    # Remove links
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)
    
    # remove the reference numbers 

    text = re.sub(r'.\d+', '.', text)

    # Remove multiple space characters
    text = re.sub('\s+',' ', text)
    
    # Convert to lowercase
    text = text.lower()
    return text

FR_speech['minutes_cleaned'] = FR_speech['minutes_cleaned'].apply(pre_process)


# Summarized the article(remove insignificant sentences)

def sum_article(text):
    from spacy.lang.en.stop_words import STOP_WORDS 
    from sklearn.feature_extraction.text import CountVectorizer 
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    doc = nlp(text)

    # create a dictionary of words and their respective frequencies 
    corpus = [sent.text.lower() for sent in doc.sents ]
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus)    
    word_list = cv.get_feature_names();    
    count_list = cv_fit.toarray().sum(axis=0)
    word_frequency = dict(zip(word_list,count_list))

    # compute the relative frequency of each word
    val=sorted(word_frequency.values())
    higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]

    # gets relative frequency of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)
        
     # Creating a ordered list (ascending order) of most important sentences
    sentence_rank={}
    for sent in doc.sents:
        for word in sent :       
            if word.text.lower() in word_frequency.keys():            
                if sent in sentence_rank.keys():
                    sentence_rank[sent]+=word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent]=word_frequency[word.text.lower()]
    top_sentences=(sorted(sentence_rank.values())[::-1])
    top_sent=top_sentences[:10]
    
    summary=[]
    for sent,strength in sentence_rank.items():  
        # print(sent)
        if strength in top_sent:
            # summary.append(sent)
            temp = ''.join(str(sent))
            summary.append(temp)
        else:
            continue

    return ' '.join(summary)

# summary0 = sum_article(FR_speech['minutes_cleaned'][0])

FR_speech['summarized_article'] = pd.Series()



for count, item in enumerate(FR_speech['minutes_cleaned']):
    FR_speech['summarized_article'][count] = sum_article(item)
    print(FR_speech['summarized_article'][count])
    
    
# cannot use apply function, the memory will just boom
FR_speech['summarized_article'] = FR_speech['minutes_cleaned'].apply(sum_article)
# FR_speech['summarized_article'][0] = FR_speech['summarized_article'][0].join() 



# Turn passage to sentences

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
FR_speech['Sentences'] = FR_speech['summarized_article'].apply(tokenizer.tokenize)

# Explode sentences into separate rows, keeping other data
FR_speech = FR_speech.explode('Sentences').reset_index(drop=True)
 

print(FR_speech['Sentences'])

# Drop the outliers

FR_speech['Sentences_length'] = pd.Series()
for count, item in enumerate(FR_speech['Sentences']):
    FR_speech['Sentences_length'][count] = len(item)
   
FR_speech['Sentences_c'] = FR_speech[ FR_speech['Sentences_length']>= 63 ]['Sentences']

FR_speech = FR_speech.dropna() 



FFR_speech = pd.read_parquet('Mfin 7036 group project/FR_speech_cleaned.parquet')

FFR_speech['Sentences_c','article_time'].reset_index()

FFR_speech_cleaned = FFR_speech.drop(columns = ['minutes','url','minutes_cleaned', 'summarized_article', 'Sentences', 'Sentences_length'])

FFR_speech_cleaned = FFR_speech_cleaned.reset_index(drop= True)
# FFR_speech_cleaned.head(10)
# ['minutes', 'url', 'article_time', 'minutes_cleaned',
#        'summarized_article', 'Sentences', 'Sentences_length', 'Sentences_c']