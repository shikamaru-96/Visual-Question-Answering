# Visual Question Answering
This is a python and keras implementation of the visual question answering model from the paper [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074).The model implemented is similar to the 2-VIS+BLSTM model mentioned in the paper except that the LSTMs are not bidirectional.This model has two image feature inputs, at the start and the end of the sentence, with different learned linear transformations. We call it 2-VIS+LSTM. 

Details about the dataset are explained at the [VisualQA website](http://www.visualqa.org/). 

## Requirements

* Python 2.7
* Numpy
* Scipy (for loading pre-computed MS COCO features)
* NLTK (for tokenizer)
* Keras(version used: 2.0.9)

## Training

* The basic usage is `python train.py`. 

* The batch size and the number of epochs can also be specified using the options `-num_epochs` and `-batch_size`. The default batch size and number of epochs are 100 and 10 respectively.

* To train with a batch size of 200 for 20 epochs, we would use: `python train.py -batch_size=200 -num_epochs=20`.

* If your device gives memory error then make swap space of 40GB and rerun the code.

## Results 
 Our model has a training accuracy of `59.70%` and validation accuracy of`52.04%`

## Pre Trained Weights 

If you don't feel like making the entire model on your machine you can download the pretrained weights from these links:

* Download the embeddings and indices from https://drive.google.com/open?id=1O73ZJtqQXOtAu8vfa_ABG9_PCixmXrT9 and save in embeddings folder

* Download the model architecture from https://drive.google.com/file/d/1GpP_0H3Tp4pWmpJh9QBVgj0UIb6Q_j-R/view save it in weights folder and rename it to model_architecture

* Download the model weights from https://drive.google.com/file/d/1pBF_sI5SaNyWNXQ2vkOBnii3WQ5NdzDM/view save it in weights folder and rename it to model_weights.

## Running the Model

* Questions can be asked on any image using the script `question_answer.py`.

* Run the script: `python question_answer.py` 

* Enter the image address in image_path (Enter n in image_path to exit) and question  `

Here are some examples of predictions:

| Image                                              | Question                   | Top Answers (left to right) |
|----------------------------------------------------|----------------------------|-----------------------------|
| <img src="examples/dog.jpeg"> 					           | Which animal is this?      | dog, cat, giraffe           |
| <img src="examples/COCO_val2014_000000000357.jpg"> | Which game is this?        | tennis, baseball, frisbee   |
| <img src="examples/COCO_val2014_000000000136.jpg"> | Which animal is this?      | giraffe, cat, bear          |
