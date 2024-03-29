import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
import scipy.io
import numpy as np

num_classes = 2
def build_pred_model(input_shape,loss_weights=None):
	model = Sequential()
	model.add(Dense(8,activation='relu',input_dim=input_shape))
	model.add(Dropout(0.05))
	model.add(Dense(4, activation='relu'))
	model.add(Dropout(0.025))
	model.add(Dense(2, activation='relu'))
	model.add(Dropout(0.025))
	model.add(Dense(num_classes, activation='softmax'))

	if loss_weights is not None:
		#not working
		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adam(),
					  metrics=['accuracy'],
					  loss_weights=loss_weights)

	else:
		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adam(),
					  metrics=['accuracy'])
	return model

def build_lstm_model(input_shape):
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(16, activation='tanh', input_shape=input_shape))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(2,activation='softmax'))
	#model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adam(),
				  metrics=['accuracy'])

	return model

def train_pred_model(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=10, verbose=1):
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=verbose,
			  validation_data=(x_test, y_test))
	return model

def evaluate_model(model,cnn_x_test,cnn_y_test,verbose=1):
	score = model.evaluate(cnn_x_test, cnn_y_test, verbose=verbose)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
