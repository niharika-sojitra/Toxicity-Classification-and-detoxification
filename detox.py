#!/usr/bin/env python
# coding: utf-8

# Install Prerequisite

# In[1]:


get_ipython().system('pip install simpletransformers datasets tqdm pandas --user')
get_ipython().system('pip install transformers sentencepiece')


# Importing Libraries

# In[2]:


import os
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import simpletransformers.t5 as st

import nltk, string
from nltk.translate.bleu_score import sentence_bleu


# Dataset Loading

# In[3]:


dataset_df = pd.read_csv('datasets/parallel_detoxification_dataset_small.tsv', sep='\t')

dataset_df.columns = ["input_text","target_text"]
dataset_df["prefix"] = "paraphrase"

dataset_df.head()


# Train Test Split

# In[4]:


train_data,test_data = train_test_split(dataset_df,test_size=0.1)


# Defininng Arguments for T5 Model

# In[5]:


args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "num_train_epochs": 4,
    "num_beams": None,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "use_multiprocessing": False,
    "save_steps": -1,
    "save_eval_checkpoints": True,
    "evaluate_during_training": False,
    "adam_epsilon": 1e-08,
    "eval_batch_size": 6,
    "fp_16": False,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0003,
    "max_grad_norm": 1.0,
    "n_gpu": 1,
    "seed": 42,
    "train_batch_size": 6,
    "warmup_steps": 0,
    "weight_decay": 0.0
}


# Defining Model

# In[6]:


model = st.T5Model("t5","s-nlp/t5-paranmt-detox", args=args, use_cuda=False)


# Train Model

# In[7]:


#model.train_model(train_data, eval_data=test_data, use_cuda=True,acc=sklearn.metrics.accuracy_score)


# In[9]:


root_dir = os.getcwd()
trained_model_path = os.path.join(root_dir,"outputs")

#arguments for saved model
args = {
"overwrite_output_dir": True,
"max_seq_length": 256,
"max_length": 50,
"top_k": 50,
"top_p": 0.95,
"num_return_sequences": 1
}

#Defining Fine-tuned Model
trained_model = st.T5Model("t5",trained_model_path,args=args, use_cuda=False)


# Model Prediction 

# In[10]:


prefix = "paraphrase"
text = 'I am tired of this school shootings by black people'
pred = trained_model.predict([f"{prefix}: {text}"])
print(pred)


# BLEU Score

# In[13]:


def BLEU(text,pred):
    text = text.split()
    for can in pred:
        candidate = can.split()
        BLEU = sentence_bleu(text, candidate)
        print('BLEU score -> {} text -> {} pred -> {}' .format(BLEU,text,candidate))


# Cosine Similarity

# In[14]:


nltk.download('punkt') 

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

#defining vectorizer
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def cosine_sim_pred(text, pred):
    for can in pred:
        cs = cosine_sim(text, can)
        print('CS score -> {} text -> {} pred -> {}' .format(cs,text,can))


# Detoxification Function

# In[15]:


def detoxification(text):
    prefix = "paraphrase"
    pred = trained_model.predict([f"{prefix}: {text}"])
    
    BLEU(text,pred)
    cosine_sim_pred(text, pred)
   
    return pred


# Testing

# In[16]:


detoxification("I will kill you")

