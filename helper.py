# Helper functions

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
########################################################################################

def test():
    return "testing"

def beautifulsoup_extract_text_fallback(response_content):
    
    '''
    This is a fallback function, so that we can always return a value for text content.
    Even for when both Trafilatura and BeautifulSoup are unable to extract the text from a 
    single URL.
    '''
    
    # Create the beautifulsoup object:
    soup = BeautifulSoup(response_content, 'html.parser')
    
    # Finding the text:
    text = soup.find_all(text=True)
    
    # Remove unwanted tag elements:
    cleaned_text = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'style',]

    # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
    # is NOT in the blacklist
    for item in text:
        if item.parent.name not in blacklist:
            cleaned_text += '{} '.format(item)
            
    # Remove any tab separation and strip the text:
    cleaned_text = cleaned_text.replace('\t', '')
    return cleaned_text.strip()
    

def extract_text_from_single_web_page(url):
    
    downloaded_url = trafilatura.fetch_url(url)
    try:
        a = trafilatura.extract(downloaded_url, output_format="json", with_metadata=True, include_comments = False,
                            date_extraction_params={'extensive_search': True, 'original_date': True})
    except AttributeError:
        a = trafilatura.extract(downloaded_url, output_format="json", with_metadata=True,
                            date_extraction_params={'extensive_search': True, 'original_date': True})
    if a:
        json_output = json.loads(a)
        return json_output['text']
    else:
        try:
            resp = requests.get(url)
            # We will only extract the text from successful requests:
            if resp.status_code == 200:
                return beautifulsoup_extract_text_fallback(resp.content)
            else:
                # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                return np.nan
        # Handling for any URLs that don't have the correct protocol
        except MissingSchema:
            return np.nan


# remove spaces, newlines
def remove_spaces(data):
    clean_text = data.replace('\\n',' ').replace('\t',' ').replace('\\',' ').replace('\n',' ')
    return clean_text

# contraction mapping
def expand_text(data):
    expanded_text = contractions.fix(data)
    return expanded_text

# handling accented characters
def handling_accented(data):
    fixed_text = unidecode(data)
    return fixed_text

def clean_data(data):
    tokens = word_tokenize(data)
    Stopword_list = get_stopword_list()
    clean_text = [word.lower() for word in tokens if (word not in punctuation) and (word.lower() not in Stopword_list) and (len(word)>2) and (word.isalpha())]
    return clean_text

# autocorrection
def autocorrection(data):
    spell = Speller(lang='en')
    corrected_text = spell(data)
    return corrected_text

# lemmatization
def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    final_data = []
    for word in data:
        lemmatized_word = lemmatizer.lemmatize(word)
        final_data.append(lemmatized_word)
    return ' '.join(final_data)


# create stopword list
def get_stopword_list():
    home_path = os.getcwd()
    stopword_path = os.path.join(home_path,r'StopWords')
    stop_word_files = os.listdir(stopword_path)
    stop_word_files = [f for f in stop_word_files if os.path.isfile(stopword_path+'/'+f)]

    path=os.path.join(stopword_path,'StopWords_Auditor.txt')
    with open(path,'r') as file:
        StopWords_Auditor = file.readlines()
        StopWords_Auditor = [text.replace('\n','') for text in StopWords_Auditor]

    path=os.path.join(stopword_path,'StopWords_Currencies.txt')
    with open(path,'r') as file:
        StopWords_Currencies = file.readlines()
        StopWords_Currencies = [text.replace('\n','').split('|') for text in StopWords_Currencies]
        StopWords_Currencies = [item.replace(' ','') for sublist in StopWords_Currencies for item in sublist]

    path=os.path.join(stopword_path,'StopWords_DatesandNumbers.txt')
    with open(path,'r') as file:
        StopWords_DatesandNumbers = file.readlines()
        StopWords_DatesandNumbers = [text.replace('\n','').split('|') for text in StopWords_DatesandNumbers]
        StopWords_DatesandNumbers = [item.replace(' ','') for sublist in StopWords_DatesandNumbers for item in sublist]

    path=os.path.join(stopword_path,'StopWords_Generic.txt')
    with open(path,'r') as file:
        StopWords_Generic = file.readlines()
        StopWords_Generic = [text.replace('\n','') for text in StopWords_Generic]

    path=os.path.join(stopword_path,'StopWords_GenericLong.txt')
    with open(path,'r') as file:
        StopWords_GenericLong = file.readlines()
        StopWords_GenericLong = [text.replace('\n','') for text in StopWords_GenericLong]

    path=os.path.join(stopword_path,'StopWords_Geographic.txt')
    with open(path,'r') as file:
        StopWords_Geographic = file.readlines()
        StopWords_Geographic = [text.replace('\n','').split('|') for text in StopWords_Geographic]
        StopWords_Geographic = [item.replace(' ','') for sublist in StopWords_Geographic for item in sublist]

    path=os.path.join(stopword_path,'StopWords_Names.txt')
    with open(path,'r') as file:
        StopWords_Names = file.readlines()
        StopWords_Names = [item.split("|")[0] for item in StopWords_Names]
        StopWords_Names = [text.replace('\n','').replace(' ','') for text in StopWords_Names]
        
    Stopword_list = []
    Stopword_list.extend(StopWords_Auditor)
    Stopword_list.extend(StopWords_Currencies)
    Stopword_list.extend(StopWords_DatesandNumbers)
    Stopword_list.extend(StopWords_Generic)
    Stopword_list.extend(StopWords_GenericLong)
    Stopword_list.extend(StopWords_Geographic)
    Stopword_list.extend(StopWords_Names)
    
    return Stopword_list

# create positive negative word list
def get_postive_list():
    home_path = os.getcwd()
    master_dict_path = os.path.join(home_path,r'MasterDictionary')

    master_dict_files = os.listdir(master_dict_path)
    master_dict_files = [f for f in master_dict_files if os.path.isfile(master_dict_path+'/'+f)]

    path=os.path.join(master_dict_path,'positive-words.txt')
    with open(path,'r') as file:
        positive_list = file.readlines()
        positive_list = [text.replace('\n','') for text in positive_list]
        
    return positive_list

def get_negative_list():
    home_path = os.getcwd()
    master_dict_path = os.path.join(home_path,r'MasterDictionary')

    master_dict_files = os.listdir(master_dict_path)
    master_dict_files = [f for f in master_dict_files if os.path.isfile(master_dict_path+'/'+f)]

    path=os.path.join(master_dict_path,'negative-words.txt')
    with open(path,'r') as file:
        negative_list = file.readlines()
        negative_list = [text.replace('\n','') for text in negative_list]
        
    return negative_list


def get_positive_score(data):
    tokens = word_tokenize(data)
    positive_dict={}
    positive_list = get_postive_list()
    for word in positive_list:
        if word in tokens:
            positive_dict[word]=+1
        else:
            positive_dict[word]=0            
    return sum(positive_dict.values())


def get_negative_score(data):
    tokens = word_tokenize(data)
    negative_dict={}
    negative_list = get_negative_list()
    for word in negative_list:
        if word in tokens:
            negative_dict[word]=-1
        else:
            negative_dict[word]=0       
    return sum(negative_dict.values())

def get_polarity_score(data):
    positive_score = get_positive_score(data)
    negative_score = get_negative_score(data) * -1
    polarity_score = ((positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001))
    return polarity_score

def get_subjectivity_score(data):
    positive_score = get_positive_score(data)
    negative_score = get_negative_score(data) * -1
    tokens = word_tokenize(data)
    total_words = len(tokens)
    subjectivity_score = (positive_score + negative_score)/ (total_words + 0.000001)
    return subjectivity_score

# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return textstatistics().syllable_count(word)

# Returns Number of Words in the text
def word_count(text):
    tokens = word_tokenize(text)
    words = len(tokens)
    return words


# Returns the number of sentences in the text
def sentence_count(text):
    sentences = sent_tokenize(text)
    return len(sentences)
               
# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length


               
# Return total Difficult Words in a text
def difficult_words(text):
    words = word_tokenize(text)
 
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()
     
    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 2:
            diff_words_set.add(word)
 
    return len(diff_words_set)

def get_diff_word_per(text):
    per_diff_words = (difficult_words(text) / word_count(text))
    return per_diff_words

def get_avg_sent_len(text):
    clean_text = remove_spaces(text)
    clean_text = expand_text(clean_text)
    clean_text = handling_accented(clean_text)
    clean_text = clean_data(clean_text)
    clean_text = lemmatization(clean_text)
    
    words = word_count(clean_text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    
    return average_sentence_length

def get_avg_words_per_sentence(text):
    clean_text = remove_spaces(text)
    clean_text = expand_text(clean_text)
    clean_text = handling_accented(clean_text)
    clean_text = clean_data(clean_text)
    clean_text = lemmatization(clean_text)
    words = len(word_tokenize(clean_text))
    sents = len(sent_tokenize(text))
    avg_words_per_sentence = words / sents
    return avg_words_per_sentence

def syllable_count(text):
    words = word_tokenize(text)
    count = 0
    vowels = "aeiouy" 
    for word in words:
        word = word.lower()
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith(("e","es","ed")):
            count -= 1
        if count == 0:
            count += 1
    return count

def count_personal_pronouns(text):
    # Use negative lookahead to ensure that the word "us" is not preceded by a period (.)
    personal_pronouns = re.findall(r"\b(I|we|my|ours|(?<!\.)us)\b", text, re.IGNORECASE)
    return len(personal_pronouns)

def get_avg_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    average_word_length = total_characters / len(words)
    return average_word_length