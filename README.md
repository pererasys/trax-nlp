# Natural Language Processing Coursework w/ Trax

Each of these files is a completed assignment in the deeplearning.ai Natural Language Specialization course hosted on Coursera. They were originally Jupyter notebooks but I've converted them to standard Python, so changes need to be made if they were to be run.

Below are descriptions of the models we created, based on my understanding and in my own words.


## Sentiment Classification

Earlier in this course I had learned about training sentiment analysis models with Naive Bayes and Logistic Regression. Classification with these models is based on the frequency of certain words in pre-classified text. The issue with this is they incorrectly classify statements like "This movie was almost good.". To solve this, we trained a neural network to identify sentiment in a more comprehensive manner.

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


## Ngram Sequences

In this assignment, we used a Recurrent Neural Network to predict the next n characters given a sequence of input tokens. To avoid vanishing or exploding gradients, this RNN has been implemented with a stack of GRU's. 

**Pre-processing**

Similar to the previous assignment, we start by preprocessing text strings and then making the conversion to tensors.

**Model architecture**

![Ngram Architecture](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/ngram_sequence_architecture.png?raw=true)

Here, we use a serial combinator to join ShiftRight, Embedding, stacked GRU, Dense, and LogSoftmax layers.

The first layer in the network is ShiftRight. What this does is shift the tensor to the right by padding the start of the tensor by n positions, in this case we shift one position to the right.

After an embedding layer converts our tensors to vectors, a stack of two GRU's is used to "remember" previous states. The key difference between GRU's and vanilla RNN's is that a GRU addresses the vanishing gradient problem by introducing gated hidden states. The gates, "update" and "reset", allow each step of the network to dispose of or retain specific pieces of information. For example, a GRU has the ability to reset previous state that may not need to be remembered, and update the new state to control how much of the previous state is copied.

![GRU Unit](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/gru_unit.png?raw=true)

After data passes through the stacked GRU's and a Dense output layer, we compute the log softmax of the outputted vectors to identify the word with the highest probability of being next in the sequence.


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

Named entity recognition is the action of extracting entities that can be found within text. To do this, we use a tag mapping for locations, people, organizations, geopolitical entities, and more. For example, the sentence "Many French citizens are going to Morocco for Christmas" contains the geopolitical entity "French", the geographic entity "Morocco", and the time indicator "Christmas". This is a very common NLP task, and in this model we use an LSTM unit to make sense of these entities in the context of a sequence of words.

**Pre-processing**

Pre-processing was mostly completed for us in this assignment. We receive our sample sentences already in tensor form, and are provided with a tag mapping which is a one-hot vector (like) dictionary of tags such as B-geo, B-gpe, B-per, etc..

**Model architecture**

![NER Architecture](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/ner_architecture.png?raw=true)

Following a similar structure to the previous models we created, we start with a serial combinator which consists of Embedding, LSTM, Dense, and LogSoftmax layers.

As in the previous models, the Embedding layer converts our tensors into vectors to be fed into the LSTM unit.

The LSTM layer in this model receives these embeddings and first decides what existing information it wants to "forget" with a sigmoid layer. To update the cell state, the output of this sigmoid layer is multiplied by the existing state. After deciding what information to throw away, the LSTM unit then decides what new information to pass to the cell state. This is done in two parts. First, a sigmoid "input layer" decides what existing cell state to update, then, a tanh layer creates a vector of new candidate values to add to the existing state. These updates and additions are then concatenated, and added to the cell state.

![LSTM unit](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/lstm_unit.png?raw=true)

The information to output through each pass of this unit is determined by another sigmoid layer, and compressed between -1 and 1 with a tanh layer.

_In the ngram sequence RNN described above, we used a variation of this LSTM unit called a GRU. A GRU is a simplified variation of the unit used in this model, essentially combining the "forget" and "input" gates into a single "update" gate, and merging the cell and hidden state._

The output of the LSTM layer is passed to a Dense output layer with n nodes, where n equals the number of tags in the tag mapping provided.

Similar to the previous models, our tag predictions are based on a final LogSoftmax layer, which gives us the probabilities of a word belonging to each tag class.


```
model = tl.Serial(
    tl.Embedding(vocab_size = vocab_size, d_feature = d_model), # Embedding layer
    tl.LSTM(n_units = d_model), # LSTM layer
    tl.Dense(n_units = len(tags)), # Dense layer with len(tags) units
    tl.LogSoftmax()  # LogSoftmax layer
)
```


## Duplicate Question Identification

Popular question-answer forums such as Quora or Stackoverflow will often provide a set of "similar" questions when you go to create a new one. In this assignment, I created a Siamese LSTM network which outputs the similarity between two input questions using cosine similarity.

**Pre-processing**

The training data used in this assignment comes from the Quora question-answer dataset. To prepare the data for training, I used NLTK to create word tokens so that I could create the vocabulary. Similar to the previous assignments, I used this vocabulary to convert questions to tensors for training and prediction.

**Model architecture**

![Siamese Architecture](https://github.com/pererasys/trax-nlp/blob/master/docs/resources/siamese_architecture.png?raw=true)

The architecture for this model is very similar to previous assignments. I first define a Serial combinator which includes Embedding, LSTM, Mean, and Normalization Function layers.

The Embedding layer converts our question tensors to vectors, which get passed to the LSTM layer, identical to the the NER model above. Where things differ is when we add a Mean layer that computes a mean vector from the LSTM output. This is then fed to a normalization function which prepares the network output for cosine similarity analysis.

This serial combination is then run in parallel to process both input questions.

```
def normalize(x):  # normalizes the vectors to have L2 norm 1
    return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))

# Processor will run on Q1 and Q2
q_processor = tl.Serial(
    tl.Embedding(vocab_size=vocab_size, d_feature=d_model), # Embedding layer
    tl.LSTM(n_units=d_model), # LSTM layer
    tl.Mean(axis=1), # Mean over columns
    tl.Fn('Normalize', lambda x: normalize(x))  # Apply normalize function
)  # Returns one vector of shape [batch_size, d_model].


# Run on Q1 and Q2 in parallel.
model = tl.Parallel(q_processor, q_processor)
```

One major difference with this model is the loss function we used for training. Where in all previous assignments we were able to take advantage of a standard CrossEntropyLoss function, determining loss when comparing two strings of text is a little bit more tricky.

Traditional triplet loss is a function of the form max(sim(A, N) - sim(A, P) + a, 0), where sim(A, N) is the similarity between an anchor and a negative example,  sim(A, P) is the similarity between an anchor and a positive example, and 'a' is the marginal offset from a loss of 0. The value 'a' allows our model to learn from time steps where the true loss was nominal or non-existent.

Our custom loss function optimizes this traditional implementation by taking the mean of two variations.

**L1 = max(closest_neg - sim(A, P) + a, 0)**\
**L2 = max(mean_neg - sim(A, P) + a, 0)**\
**L = mean(L1, L2)**

```
def TripletLossFn(v1, v2, margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        jax.interpreters.xla.DeviceArray: Triplet Loss.
    """
    
    # use fastnp to take the dot product of the two batches
    scores = fastnp.dot(v1, fastnp.transpose(v2))  # pairwise cosine sim
    
    # calculate new batch size
    batch_size = len(scores)
    
    # use fastnp to grab all postive `diagonal` entries in `scores`
    positive = fastnp.diagonal(scores)  # the positive ones (duplicates)
    
    # multiply `fastnp.eye(batch_size)` with 2.0 and subtract it out of `scores`
    negative_without_positive = scores - fastnp.eye(batch_size)
    
    # take the row by row `max` of `negative_without_positive`.
    closest_negative = negative_without_positive.max(axis=[1])
    
    # subtract `fastnp.eye(batch_size)` out of 1.0 and do element-wise multiplication with `scores`
    negative_zero_on_duplicate = (1.0 - fastnp.eye(batch_size)) * scores
    
    # use `fastnp.sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)` 
    mean_negative = fastnp.sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    
    # compute `fastnp.maximum` among 0.0 and `A`
    # A = subtract `positive` from `margin` and add `closest_negative` 
    triplet_loss1 = fastnp.maximum((margin - positive + closest_negative), 0.0)
    
    # compute `fastnp.maximum` among 0.0 and `B`
    # B = subtract `positive` from `margin` and add `mean_negative`
    triplet_loss2 = fastnp.maximum((margin - positive + mean_negative), 0.0)
    
    # add the two losses together and take the `fastnp.mean` of it
    triplet_loss = fastnp.mean(triplet_loss1 + triplet_loss2)
    
    return triplet_loss
```

## Neural Machine Translation

TODO
