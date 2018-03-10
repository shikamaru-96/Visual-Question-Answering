import numpy as np
import prepare_data
import models
import argparse
import sys

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_epochs', type=int, default=10)
	parser.add_argument('-batch_size', type=int, default=100)
	args = parser.parse_args()

	print('Loading questions ...')
	questions_train = prepare_data.get_questions_matrix('train')
	questions_val = prepare_data.get_questions_matrix('val')
	print('Loading answers ...')
	answers_train = prepare_data.get_answers_matrix('train')
	answers_val = prepare_data.get_answers_matrix('val')
	print('Loading image features ...')
	img_features_train = prepare_data.get_coco_features('train')
	img_features_val = prepare_data.get_coco_features('val')
	print('Creating model ...')
	
	model = models.vis_lstm_2()
	X_train = [img_features_train, questions_train, img_features_train]
	X_val = [img_features_val, questions_val, img_features_val]


	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	model.fit(X_train,answers_train,
		nb_epoch=args.num_epochs,
		batch_size=args.batch_size,
		validation_data=(X_val,answers_val),
		verbose=1)
	model.save_weights('weights/model_weights.h5')
	with open('weights/model_architecture.json', 'w') as f:
	        f.write(model.to_json())


if __name__ == '__main__':main()

