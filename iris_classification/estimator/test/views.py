# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:35:25 2019

@author: mehdi_jafari
"""

from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# keras for loading the model 
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
# other utilities
import urllib.parse
import json
import numpy as np
import os
import pickle
import logging
import re

########################
########################
########################
########################
########################

########################
########################
########################
########################
########################


dirpath = os.getcwd()
sys.path.append(dirpath+"/polls")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import keras_model
from .keras_model import deepmoji_architecture
from .keras_model import config
###############################
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

global graph
# graph = tf.get_default_graph() 

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
###############################
persian_alpha_codepoints = '\u0621-\u0628\u062A-\u063A\u0641-\u0642\u0644-\u0648\u064E-\u0651\u0655\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC'
space_codepoints ='\u0020\u2000-\u200F\u2028-\u202F'
additional_arabic_characters_codepoints = '\u0629\u0643\u0649-\u064B\u064D\u06D5'
# eliminate illegal chars from the string 
def remove_non_persian(string):
    if type(string)!=type("string"):
        string = str(string)
    string = re.sub("[^"+persian_alpha_codepoints+space_codepoints+additional_arabic_characters_codepoints+"]+", "", string)
    return string
def remove_repeatative(string):
    if type(string)!=type("string"):
        string = str(string)
    repeat_pattern = re.compile(r'(.)\1*')
    match_substitution = r'\1'
    string = repeat_pattern.sub(r'\1', string)
    return string 
###############################

# globals
global model# = None
global tokenizer# = None
global graph
global loaded_model
global LANG

tokenizer_path = dirpath+"/polls/models/tokenizer.pkl"
model_path = dirpath+"/polls/models/model.h5"
loaded_model = ''
MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, NB_CLASS, LANG = \
    config(dirpath+"/polls/models/config.txt")



model = None
tokenizer = None 


# load model for using
def model_load():
    sess = get_session()
    ktf.set_session(sess)
    global model
    global graph

    global model_path
    try:
        model = deepmoji_architecture(NB_CLASS, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, feature_output=False, 
                                    embed_dropout_rate=0.4, final_dropout_rate=0.4, 
                                    embed_l2=0, return_attention=False)
        model.load_weights(model_path)
    except Exception as e:
        raise e
    graph = tf.get_default_graph()
    return model, graph


# padding the input string and changing it into tensor
def sentences2tensor(sentences):
    global tokenizer
    sequences = tokenizer.texts_to_sequences([sentences])
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    data_post = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post', truncating='post')
    return [data, data_post]

# load tokenizer module
def load_tokenizer():
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        raise e
    return tokenizer

def detect(string, graph):
    global model
    tensor_string = sentences2tensor(string)
    with graph.as_default():
        result = model.predict(tensor_string)
    return result

try:
    model, graph = model_load()
    tokenizer = load_tokenizer()
    loaded_model = ''
except:
    raise
    
@csrf_exempt
def index(request):

    global model
    global tokenizer
    global model_path
    global graph
    global loaded_model
    global LANG
    response = []
    json_params=request.POST.get('param')
    
    try:
        params = json.loads(json_params,strict=False)
    except:
        raise Exception({"errorCode":"201","errorMessage":"Invalid JSON Format","systemErrorMessage":"system error"})
    lang =  urllib.parse.unquote_plus(params['lang']) 
    
    if lang !=LANG:
        # final_topics={"unknown language"}
        raise Exception({"errorCode":"202","errorMessage":"Invalid Language","systemErrorMessage":"system error"})
    
    
    else:
        body_news= urllib.parse.unquote_plus(params['docBody'])                               
        if body_news.strip() == "":
            segment_polarity={"segment":"", "polarity":"", "polarityScore":"", "emotion":"", "emotionScore":"", "aspect":""}
            response.append(segment_polarity) 
            return HttpResponse(str({"segmentsList":response}))
        # body_news = remove_non_persian(remove_repeatative(body_news))
        sentences = body_news.strip().split(".")
        sentences = [remove_non_persian(remove_repeatative(item)) for item in sentences]
        # print(sentences)
        results = []
        for sentence in sentences:
            result = detect(sentence, graph)
            result = result[0].tolist()
            results.append(result)
            # print(results)
        result = []
        for sentence, score in zip(sentences, results):
            segment_insult={"segment":sentence, 
                            "classes":  [

                            ["blasphemy", score[0]], 
                            ["IRIInsult", score[1]], 
                            ["Insult", score[2]], 
                            ["Immoral", score[3]], 
                            ["safe", score[4]],

                                     ] 
                           }                                 
            result.append(segment_insult)

    return JsonResponse({"segmentsList":result})
    # return  JsonResponse({"segmentsList":response})



