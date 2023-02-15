# Blackcoffer
# Data Extraction and NLP
# Test Assignment
###############################################################################################
#Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import json
import numpy as np
import requests
from requests.models import MissingSchema
import trafilatura

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import contractions
from unidecode import unidecode
from autocorrect import Speller

import os
from textstat.textstat import textstatistics
import re

import warnings
warnings.filterwarnings('ignore')

import helper as hp
###############################################################################################

# Data Extraction
df = pd.read_excel(io='Input.xlsx')
df['Extracted_Text'] = df['URL'].apply(hp.extract_text_from_single_web_page)

# dropping null values where the url was not successfully connected
df.dropna(inplace=True)

#Storing extracted data in Text files
path_text_files = 'Extracted_text/'

for id in df['URL_ID']:
    text = df[df['URL_ID']==id]['Extracted_Text'].iloc[0]
    with open (f'{path_text_files}/{id}.txt', 'w', encoding='utf-8') as file:  
        file.write(text)  

# 1 Sentimental Analysis
# 1.1 Cleaning using Stop Words Lists

clean_text = df['Extracted_Text'].apply(hp.remove_spaces)
clean_text = clean_text.apply(hp.expand_text)
clean_text = clean_text.apply(hp.handling_accented)
clean_text = clean_text.apply(hp.clean_data)
clean_text = clean_text.apply(hp.lemmatization)

df['clean_text'] = clean_text

# 1.2 Creating a dictionary of Positive and Negative words
# positive_word_list, negative_word_list = hp.get_postive_negative_list()

# 1.3 Extracting Derived variables
# ***Positive Score: This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.
# ***Negative Score: This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. We multiply the score with -1 so that the score is a positive number.
# ***Polarity Score: This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: 
# Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
# Range is from -1 to +1
# ***Subjectivity Score: This is the score that determines if a given text is objective or subjective. It is calculated by using the formula: 
# Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
# Range is from 0 to +1

df['POSITIVE SCORE'] = df['clean_text'].apply(hp.get_positive_score)
df['NEGATIVE SCORE'] = df['clean_text'].apply(hp.get_negative_score) * -1
df['POLARITY SCORE'] = df['clean_text'].apply(hp.get_polarity_score)
df['SUBJECTIVITY SCORE'] = df['clean_text'].apply(hp.get_subjectivity_score)

# 2 Analysis of Readability
# Analysis of Readability is calculated using the Gunning Fox index formula described below.
# Average Sentence Length = the number of words / the number of sentences
# Percentage of Complex words = the number of complex words / the number of words 
# Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)

df['AVG SENTENCE LENGTH'] = df['Extracted_Text'].apply(hp.get_avg_sent_len)
df['PERCENTAGE OF COMPLEX WORDS'] = df['clean_text'].apply(hp.get_diff_word_per)
df['FOG INDEX'] = 0.4 * (df['AVG SENTENCE LENGTH'] + df['PERCENTAGE OF COMPLEX WORDS'])

# 3 Average Number of Words Per Sentence
# The formula for calculating is:
# Average Number of Words Per Sentence = the total number of words / the total number of sentences

df['AVG NUMBER OF WORDS PER SENTENCE'] = df['Extracted_Text'].apply(hp.get_avg_words_per_sentence)

# 4 Complex Word Count
# Complex words are words in the text that contain more than two syllables.

df['COMPLEX WORD COUNT'] = df['clean_text'].apply(hp.difficult_words)

# 5 Word Count
# We count the total cleaned words present in the text by 
# 1.	removing the stop words (using stopwords class of nltk package).
# 2.	removing any punctuations like ? ! , . from the word before counting.

df['WORD COUNT'] = df['clean_text'].apply(hp.word_count)

# 6 Syllable Count Per Word
# We count the number of Syllables in each word of the text by counting the vowels present in each word. We also handle some exceptions like words ending with "es","ed" by not counting them as a syllable.

df['SYLLABLE PER WORD'] = df['clean_text'].apply(hp.syllable_count)

# 7 Personal Pronouns
# To calculate Personal Pronouns mentioned in the text, we use regex to find the counts of the words - “I,” “we,” “my,” “ours,” and “us”. Special care is taken so that the country name US is not included in the list.

df['PERSONAL PRONOUNS'] = df['Extracted_Text'].apply(hp.count_personal_pronouns)

# 8 Average Word Length
# Average Word Length is calculated by the formula:
# Sum of the total number of characters in each word/Total number of words

df['AVG WORD LENGTH'] = df['clean_text'].apply(hp.get_avg_word_length)

# 4 Output Data Structure

df_new = df[['URL_ID', 'URL', 'POSITIVE SCORE',
       'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
       'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
       'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
       'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']]

df_new.to_excel('Output.xlsx',index=False)