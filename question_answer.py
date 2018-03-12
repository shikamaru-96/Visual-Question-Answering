import numpy as np
import embedding as ebd
import prepare_data
import models
import argparse
import sys
import keras.backend as K
from nltk import word_tokenize
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json

def extract_image_features(img_path):
	model = models.VGG_16('weights/vgg16_weights_th_dim_ordering_th_kernels.h5')
	img = image.load_img(img_path,target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)
	last_layer_output = K.function([model.layers[0].input,K.learning_phase()],
		[model.layers[-1].input])
	features = last_layer_output([x,0])[0]
	return features

def preprocess_question(question):
	word_idx = ebd.load_idx()
	tokens = word_tokenize(question)
	seq = []
	for token in tokens:
		seq.append(word_idx.get(token,0))
	seq = np.reshape(seq,(1,len(seq)))
	return seq

def main():
	with open('weights/model_architecture.json', 'r') as f:
    		model = model_from_json(f.read())
   	model.load_weights('weights/model_weights.h5')
	
	img_path = "o"
	question = "o"

	while img_path != "n":	
		img_path = raw_input("Enter image path: ")
		if img_path == "n":
			continue
		question = raw_input("Enter question: ")
		try:
			img_features = extract_image_features(img_path)
		except:
			print("Invalid image path\n")
			continue
		seq = preprocess_question(question)
		x = [img_features, seq, img_features]
		probabilities = model.predict(x)[0]
		answers = np.argsort(probabilities[:1000])
		top_answers = [prepare_data.top_answers[answers[-1]],
			prepare_data.top_answers[answers[-2]],
			prepare_data.top_answers[answers[-3]]]
		print('Top answers: %s, %s, %s.' % (top_answers[0],top_answers[1],top_answers[2]))
		print("\n")

if __name__ == '__main__':main()
