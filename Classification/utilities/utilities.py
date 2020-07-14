import csv
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf


######################################################
#   Word Matrix and vocabulary build                 #
######################################################
def flatten(list):
    '''
    Helper function that flattens 2D lists.
    '''
    return [i for sublist in list for i in sublist]


wnut_b = {'B-corporation': 12,
          'B-creative-work': 11,
          'B-group': 10,
          'B-location': 9,
          'B-person': 8,
          'B-product': 7,
          'I-corporation': 6,
          'I-creative-work': 5,
          'I-group': 4,
          'I-location': 3,
          'I-person': 2,
          'I-product': 1,
          'O': 0}

case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3,
            'other': 4, 'mainly_numeric': 5, 'contains_digit': 6, 'PADDING_TOKEN': 7}


def learn_embedding(vocab):
    '''
    Creating Global vectors weight matrix, later might be used for initializing the word embeddings layer
    '''
    embeddings_index = dict()
    file = open('data/glove.twitter.27B.200d.txt', 'r', encoding='utf8')
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    file.close()

    embedding_matrix = numpy.zeros((len(vocab) + 1, 200))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

            return embedding_matrix


def create_lookup(sentences, voc):
    """
    This function .....
    :param sentences:
    :param voc:
    :return:
    """
    words = []
    chars = []
    labels = []

    for sentence in sentences:
        for word_label in sentence:
            words.append(word_label[0])
            labels.append(word_label[1])

            for char in word_label[0]:
                chars.append(char)
    word_counts = Counter(words)
    vocb_inv = [x[0] for x in word_counts.most_common()]
    vocb = {x: i + 1 for i, x in enumerate(vocb_inv)}
    vocb['PAD'] = 0
    id_to_vocb = {i: x for x, i in vocb.items()}

    char_counts = Counter(chars)
    vocb_inv_char = [x[0] for x in char_counts.most_common()]
    vocb_char = {x: i + 1 for i, x in enumerate(vocb_inv_char)}

    labels_counts = Counter(labels)

    labelVoc_inv, labelVoc = label_index(labels_counts, voc)

    return [vocb_char, labelVoc]


def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


def create_sequences(sentences, vocab_char, labelVoc, word_maxlen, sent_maxlen):
    '''
        This function is used to pad the word into the same length.
    '''
    x = []
    x_w = []
    y = []
    img_features = []
    for sentence in sentences:
        w_id = []
        w = []
        y_id = []
        for word_label in sentence:
            w_id.append(word_label[0])
            # w.append(vocabulary[word_label[0]])

            y_id.append(labelVoc[word_label[1]])
        x.append(w_id)
        # x_w.append(w)
        y.append(y_id)
    print(y)
    # print(y)

    y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sent_maxlen, padding="post")
    # x_word = tf.keras.preprocessing.sequence.pad_sequences(x_w, maxlen=sent_maxlen, padding="post")
    '''
    if image_features != None:
        img_x = np.asarray(image_features)
        img_features.append(img_x)
    else:
        pass
'''
    x_c = []
    for sentence in sentences:
        s_pad = np.zeros([sent_maxlen, word_maxlen], dtype=np.int)
        s_c_pad = []
        for word_label in sentence:
            w_c = []
            char_pad = np.zeros([word_maxlen], dtype=np.int)
            for char in word_label[0]:
                w_c.append(vocab_char[char])
            if len(w_c) <= word_maxlen:
                char_pad[:len(w_c)] = w_c
            else:
                char_pad = w_c[:word_maxlen]

            s_c_pad.append(char_pad)

        for i in range(len(s_c_pad)):
            s_pad[sent_maxlen - len(s_c_pad) + i, :len(s_c_pad[i])] = s_c_pad[i]

        x_c.append(s_pad)

    # building cases

    addChar = []
    #    np.zeros(sent_maxlen, word_maxlen)
    for sentence in sentences:
        cased_word = []
        for word_label in sentence:
            ortho = getCasing(word_label[0], case2Idx)
            cased_word.append(ortho)

        addChar.append(cased_word)
    return [x, y, x_c, addChar]


def label_index(labels_counts, labelVoc):
    '''
       the input is the output of Counter. This function defines the (label, index) pair,
       and it cast our datasets label to the definition (label, index) pair.
    '''

    num_labels = len(labels_counts)
    labelVoc_inv = [x[0] for x in labels_counts.most_common()]
    if len(labelVoc) < num_labels:
        for key, value in labels_counts.items():
            if not key in labelVoc:
                labelVoc.setdefault(key, len(labelVoc))
    return labelVoc_inv, labelVoc


######################################################
#   Prediction files prepration                      #
######################################################
def write_file(filename, dataset, delimiter='\t'):
    """dataset is a list of tweets where each token can be a tuple of n elements"""
    with open(filename, '+w', encoding='utf8') as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE, quotechar='')

        for tweet in dataset:
            writer.writerow(list(tweet))


def save_predictions(filename, tweets, labels, predictions):
    """save a file with token, label and prediction in each row"""
    dataset, i = [], 0
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, labels[n], predictions[n]))
        dataset += tweet_data + [()]
    write_file(filename, dataset)


def getLabels(y_test, vocabulary):
    '''
    Maps integer to the label map
    '''
    #
    classes = []
    # y = np.array(y_test).tolist()
    for i in y_test:
        label = []
        pre = [[k for k, v in vocabulary.items() if v == j] for j in i]
        for i in pre:
            for j in i:
                label.append(j)
        classes.append(label)
    return classes


######################################################
#   Training history plotting                        #
######################################################


def show_training_loss_plot(hist):
    '''
    Graph plotter
    '''

    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
    plt.plot(range(len(train_loss)), val_loss, color="blue", label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.show()
