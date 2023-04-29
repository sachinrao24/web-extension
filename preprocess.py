import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re, string
import demoji
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

import torch
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer


class Preprocess():

    def remove_emoji(text):
        return demoji.replace(text, '')
    
    def strip_all_entities(text): 
        text = text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
        text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
        text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
        banned_list= string.punctuation
        table = str.maketrans('', '', banned_list)
        text = text.translate(table)
        text = [word for word in text.split() if word not in stop_words]
        text = ' '.join(text)
        text =' '.join(word for word in text.split() if len(word) < 14) # remove words longer than 14 characters
        return text

    #remove contractions
    def decontract(text):
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text
    
    #clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the "#" symbol
    def clean_hashtags(tweet):
        new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
        new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove # symbol from words in the middle of the sentence
        return new_tweet2
    
    #Filter special characters such as "&" and "$" present in some words
    def filter_chars(a):
        sent = []
        for word in a.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        return ' '.join(sent)

    #Remove multiple sequential spaces
    def remove_mult_spaces(text):
        return re.sub("\s\s+" , " ", text)

    #Stemming
    def stemmer(text):
        tokenized = nltk.word_tokenize(text)
        ps = PorterStemmer()
        return ' '.join([ps.stem(words) for words in tokenized])

    #Lemmatization 
    #NOTE:Stemming seems to work better for this dataset
    def lemmatize(text):
        tokenized = nltk.word_tokenize(text)
        lm = WordNetLemmatizer()
        return ' '.join([lm.lemmatize(words) for words in tokenized])

    #Then we apply all the defined functions in the following order
    def deep_clean(text):
        text = remove_emoji(text)
        text = decontract(text)
        text = strip_all_entities(text)
        text = clean_hashtags(text)
        text = filter_chars(text)
        text = remove_mult_spaces(text)
        text = stemmer(text)
        return text
    
    def bert_tokenizer(data):
        MAX_LEN = 128
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids = []
        attention_masks = []
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]` special tokens
                max_length=MAX_LEN,             # Choose max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length 
                return_attention_mask=True      # Return attention mask
                )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks
        
