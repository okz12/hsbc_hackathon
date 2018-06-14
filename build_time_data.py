import numpy as np
#assume data is in order
def construct_time_series_inputs(X_train,lookback):

	time_series_x_train = []
	for i in range(len(X_train)):
		time_series_x_train.append(X_train[max(0,i-lookback):i+1].tolist())

	return time_series_x_train