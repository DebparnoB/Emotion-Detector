from flask import Flask, make_response, request
from flask_cors import CORS, cross_origin

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
from keras import backend as K


application = Flask(__name__)
CORS(application)
cors = CORS(application, resources = {
    r"/*": {
        "origins": "*"
    }
})

best_model =  load_model('BalanceNet_T20.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@application.route('/')
def index():
    return 'Application Works!'

@application.route('/process', methods= ['POST'])
def process():
    
    # show the user profile for that user
    firstName = request.form['firstName']
    	
    MAX_SEQUENCE_LENGTH = 30
    result = ""
    text = ["" for _ in range(5)]
    text[0] = str(firstName)
    sequences_test = tokenizer.texts_to_sequences(text)
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    	
    y_prob = best_model.predict(data_test)
    result = str(y_prob[0][0]) + ',' + str(y_prob[0][1]) +',' + str(y_prob[0][2]) + ',' + str(y_prob[0][3]) + ',' + str(y_prob[0][4])
    	
    return result
	
    
if(__name__=='__main__'):
    from waitress import serve
    serve(application, host='0.0.0.0', port=8080)
