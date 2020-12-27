# Natural Language Processing Coursework w/ Trax

Each of these files is a completed assignment in the deeplearning.ai Natural Language Specialization course hosted on Coursera. They were originally Jupyter notebooks but I've converted them to standard Python, so changes need to be made if they were to be run.


## Sentiment Analysis

Earlier in this course I had learned about training sentiment analysis models with Naive Bayes and Logistic Regression. Classification with these models is based on the frequency of certain words in pre-classified text. The issue with this is they incorrectly classify statements like "This movie was almost good.". To solve this, we trained a neural network to learn the intricacies of context and sentiment.

**Pre-processing**

Pre-process tweets by removing stop words and unwanted characters such as "#", then convert the string to a list of words. Store each word as a key in a vocabulary dictionary which's value is an autoincremented integer. This dictionary also includes indexes for unknown tokens, padding tokens, and end of sentence tokens. 

```
"I am happy!" -> ["i", "am", "happy"] -> {"i": 0, "am": 1, "happy": 2, ...}
```

The previous step is done so that the model is able to classify text in a mathematical way. To do this, each tweet is converted to a "tensor", which is just a numerical representation of the text. To train the model, we create three Python generators (training, validation, and testing) which return a subset of positive and negative examples, the corresponding labels for the subset, and an array of weights which specify the importance of each example. 

```
Tweet-to-tensor

"I am happy!" -> ["i", "am", "happy"] -> [0, 1, 2]
```


**Model architecture**

![Classifier Architecture](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/sentiment_architecture.jpg?raw=true)

We use a serial cominator to combine embedding, mean, dense, and log softmax layers.

The Embedding layer converts our tweet tensors to vectors, which are fed to a hidden Mean layer that calculates an embedding vector which is an average of all words in the vocabulary. This allows us to minimize the number of parameters passed to the dense layer, making training more efficient.

The Dense layer is our "trainable" layer, it computes the proper parameters for the function y = Wx + b. The parameters to "learn" are the weight matrix, W, and a bias vector, b. Each node (in this case 2) takes the outputs from the Mean layer and updates these parameters based on the loss calculated by a cross entropy function.

To properly train our model, we then compute the softmax of the output vector, y. The softmax function turns the vector of real values into a vector of probabilities which sum to 1. Here, we use a LogSoftmax layer, which takes the softmax and then computes the base e log of the probabilities. For this classifier, using log probabilities ensures that the cross entropy loss function will treat incorrect classifications more harshly, which creates a better training environment for the dense layer.

Here are the important bits of the model implementation:
```
embed_layer = tl.Embedding(
    vocab_size=vocab_size,  # Size of the vocabulary
    d_feature=embedding_dim  # Embedding dimension
)

mean_layer = tl.Mean(axis=1)

dense_output_layer = tl.Dense(n_units=output_dim)   # define the output dimensions (number of nodes)

log_softmax_layer = tl.LogSoftmax()

model = tl.Serial(
    embed_layer,
    mean_layer,
    dense_output_layer,
    log_softmax_layer
)
```



## Ngram Generation w/ RNN

In this assignment, we used a Recurrent Neural Network to predict the next n characters given a sequence of input tokens. To avoid vanishing or exploding gradients, this RNN has been implemented with a stack of GRU's. 

**Pre-processing**

Similar to the previous assignment, we start by preprocessing text strings and then making the conversion to tensors.

**Model architecture**

![Deep n-gram Architecture](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/ngram_model.png?raw=true)

Here, we use a serial combinator to join ShiftRight, Embedding, stacked GRU, Dense, and LogSoftmax layers.

The first layer in the network is ShiftRight. What this does is shift the tensor to the right by padding the start of the tensor by n positions, in this case we shift one position to the right.

After an embedding layer converts our tensors to vectors, a stack of two GRU's is used to "remember" previous states. The key difference between GRU's and vanilla RNN's is that a GRU addresses the vanishing gradient problem by introducing gated hidden states. The gates, "update" and "reset", allow each step of the network to dispose of or retain specific pieces of information. For example, a GRU has the ability to reset previous state that may not need to be remembered, and update the new state to control how much of the previous state is copied.

After data passes through the stacked GRU's and a Dense output layer, we compute the log softmax of the outputted vectors to identify the word with the highest probability of being next in the sequence.


Here are the important bits of the model implementation:
```
model = tl.Serial(
  tl.ShiftRight(mode=mode), # Stack the ShiftRight layer
  tl.Embedding(vocab_size=vocab_size, d_feature=d_model), # Stack the embedding layer
  [tl.GRU(n_units=d_model) for i in range(n_layers)], # Stack GRU layers of d_model units
  tl.Dense(n_units=vocab_size),
  tl.LogSoftmax()
)
```

## Named Entity Recognition

TODO


## Duplicate Question Classification

TODO


## Language Translation

TODO
