
import json


with open('config.json') as config_file:
    configs = json.load(config_file)
intents_path = configs['intents']
words_path = configs['words']
classes_path = configs['classes']
model_path = configs['saved_model']

