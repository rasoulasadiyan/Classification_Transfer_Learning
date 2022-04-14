from tensorflow import keras
import numpy as np
import keras.models
from keras.models import model_from_json
# from scipy.misc import imread, imresize,imshow
import scipy

# import tensorflow as tf


print(scipy.__version__)

def init(): 
	try:
		json_file = open('static/model/model.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)

		#load weights into new model
		model_weight_path='static/model/model.h5'
		loaded_model.load_weights(model_weight_path)
		print("loaded model from disk")

	except Exception as ex:
		print('Error!!!\nThe model could not be loaded')

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	# graph = tf.get_default_graph()

	return loaded_model