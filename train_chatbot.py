import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
import random
from model import create_and_train_model
import config_reader

words=[]
classes=[]
documents=[]
ignore_words=['?', '!']

data = open(config_reader.intents_path).read()
intents = json.loads(data)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append(w, intent['tag'])
        # add classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower() for w in words if w not in ignore_words)]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

print(len(words), "Unique Words: ", words)

pickle.dump(words, open(config_reader.words_path, 'wb'))
pickle.dump(classes, open(config_reader.classes_path, 'wb'))


training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    # lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower() for word in pattern_words)]
    # bag of words algorithm
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag, output_row)

    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    training_x = list(training[:, 0])
    training_y = list(training[:, 1])
    print("Training data created")


model = create_and_train_model(training_x, training_y)



















