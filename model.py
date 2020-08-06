# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np
import config_reader




def create_and_train_model(train_data, classes):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_data),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    # Compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #fitting and saving the model
    trained_model = model.fit(np.array(train_data), np.array(classes),
              epochs=200, batch_size=5, verbose=1)
    model.save(config_reader.model_path, trained_model)

    print('model created...')
