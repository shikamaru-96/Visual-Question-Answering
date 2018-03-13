## Instructions for preparing embeddings

Download and extract the pretrained common crawl 300D word vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip.
Use the script `embedding.py` to generate the embedding matrix and word indices. The usage is as follows:

```
$ python embedding.py -address address-of-extracted-glove-file
```
You can also download the prebuilt embedding matrix and word indices from here : https://drive.google.com/open?id=1O73ZJtqQXOtAu8vfa_ABG9_PCixmXrT9
