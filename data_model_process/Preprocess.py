from collections import Counter
import pickle
import pandas as pd
import numpy as np

import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import wordcloud

# Defining a function to clean up the text
def step_1_clean(Text):
    sms = re.sub('[^a-zA-Z]', ' ', Text) #Replacing all non-alphabetic characters with a space
    sms = sms.lower() #converting to lowecase
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

def step_2_tokenize(Text):
    return nltk.word_tokenize(Text)

# Removing the stopwords function
def step_3_remove_stopwords(Text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in Text if word not in stop_words]
    return filtered_text

# lemmatize string
def step_4_lemmatizer(Text):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in Text]
    return lemmas

# Creating a corpus of text feature to encode further into vectorized form
def step_5_join_text(Text):
    return " ".join(Text)

# Changing text data in to numbers. 
def step_6_numerize(Text, path_to_tfidf):
    tfidf = pickle.load(open(path_to_tfidf, 'rb')) # "../" + 
    return tfidf.transform(np.asarray([Text]))

# Function for DataFrame Only
def step_all_for_data_frame(data, path_to_tfidf):
    # step 1
    data["Clean_Text"] = data["Text"].apply(step_1_clean)
    
    # step 2
    data["Tokenize_Text"]=data.apply(lambda row: nltk.word_tokenize(row["Clean_Text"]), axis=1)
    
    # step 3
    data["Nostopword_Text"] = data["Tokenize_Text"].apply(step_3_remove_stopwords)
    
    # step 4
    data["Lemmatized_Text"] = data["Nostopword_Text"].apply(step_4_lemmatizer)
    
    # step 5
    corpus= []
    for i in data["Lemmatized_Text"]:
        msg = ' '.join([row for row in i])
        corpus.append(msg)
    data["corpus"] = corpus
    
    tfidf = pickle.load(open("../" + path_to_tfidf, 'rb'))
    X = tfidf.transform(np.asarray(corpus))
    return data, X    

def get_dict_count(data, class_of_label = 1):
    ''' Class of label equal `1` is Spam
    
        `0` is Ham
    '''
    text = []
    for elm in data[data["Class"] == class_of_label]["Text"].values:
        for toz in elm.split():
            text.append(toz.strip())
    counter = Counter(text)
    common_words = counter.most_common(15)

    words = []
    values = []
    for key, value in dict(common_words).items():
        words.append(key)
        values.append(value)

    return pd.DataFrame({"Word" : words, "Count": values}).sort_values(by=['Count'], ascending=False)