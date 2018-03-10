# Visual Question Answering
This is a python and keras implementation of the visual question answering model. Consider the paper [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074).The model implemented is similar to the 2-VIS+BLSTM model from the paper mentioned above except that the LSTMs are not bidirectional.This model has two image feature inputs, at the start and the end of the sentence, with different learned linear transformations. We call it 2-VIS+LSTM. 

Details about the dataset are explained at the [VisualQA website](http://www.visualqa.org/). 

## Requirements

* Python 2.7
* Numpy
* Scipy (for loading pre-computed MS COCO features)
* NLTK (for tokenizer)
* Keras
* Theano

## Training

* The basic usage is `python train.py`. 

* The batch size and the number of epochs can also be specified using the options `-num_epochs` and `-batch_size`. The default batch size and number of epochs are 100 and 10 respectively.

* To train with a batch size of 200 for 20 epochs, we would use: `python train.py -batch_size=200 -num_epochs=20`.


## Model (2-VIS+LSTM)

<img src="examples/model.png">

##Performance

The model gives a validation accuracy of 54.01% after 10 epochs keeping batch size=200.


## Prediction

* Q&A can be performed on any image using the script `question_answer.py`.

* The options `-question` and `-image` are used to specify the question and address of the image respectively. 

* An example of usage is: `python question_answer.py -image="examples/COCO_val2014_000000000136.jpg" -question="Which animal is this?"`

Here are some examples of predictions:

| Image                                              | Question                   | Top Answers (left to right) |
|----------------------------------------------------|----------------------------|-----------------------------|
| <img src="examples/COCO_val2014_000000000136.jpg"> | Which animal is this?      | giraffe, cat, bear          |
| <img src="examples/COCO_val2014_000000000073.jpg"> | Which vehicle is this?     | motorcycle, taxi, train     |
| <img src="examples/COCO_val2014_000000000196.jpg"> | How many dishes are there? | 5, 3, 2                     |
| <img src="examples/COCO_val2014_000000000283.jpg"> | What is in the bottle?     | water, beer, wine           |
| <img src="examples/COCO_val2014_000000000357.jpg"> | Which sport is this?       | tennis, baseball, frisbee   |
