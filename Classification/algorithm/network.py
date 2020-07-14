import tensorflow as tf
import tensorflow_hub as hub
from keras import Input
from keras import backend as K, Model
from keras.engine import Layer
from keras.initializers import RandomUniform
from keras.layers import Embedding, Reshape, TimeDistributed, Conv1D, MaxPooling1D, regularizers, np, Flatten, Dropout, \
    Bidirectional, LSTM, add, merge, Dense

from utilities.setting import *


def network_model(sent_maxlen, word_maxlen, char_vocab, dataset_type, architecture=None):
    if architecture == BASE_MODEL:
        model = build_bilstm_cnn_model(sent_maxlen, word_maxlen, char_vocab, dataset_type)
        return model
    else:
        pass


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(inputs={
            "tokens": tf.squeeze(tf.cast(x, "string")),
            "sequence_len": tf.constant(100 * [100])
        },
            signature="tokens",
            as_dict=True)["elmo"]
        return result

        # def compute_mask(self, inputs, mask=None):
        # return K.not_equal(inputs, '__PAD__')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 100, self.dimensions)


def getCharCNN(sent_maxlen, word_maxlen, char_vocab_size):
    '''
    Character_level CNN for character representations based on mentioned citation mentioned in the paper, however,
     modified to our case
    '''
    char_out_dim = 30
    char_input = Input(shape=(sent_maxlen, word_maxlen))

    char_embed_layer = Embedding(input_dim=char_vocab_size,
                                 output_dim=char_out_dim,
                                 input_length=(sent_maxlen, word_maxlen,),
                                 embeddings_initializer=RandomUniform(minval=-np.sqrt(3 / char_out_dim),
                                                                      maxval=np.sqrt(3 / char_out_dim)))(char_input)
    # dropout = Dropout(0.5)(char_in)
    c_reshape = Reshape((sent_maxlen, word_maxlen, 30))(char_embed_layer)
    conv1d_out = TimeDistributed(Conv1D(kernel_size=3,
                                        filters=30,
                                        padding='same',
                                        activation='tanh',
                                        strides=1,
                                        kernel_regularizer=regularizers.l2(0.001)))(c_reshape)
    maxpool_out = TimeDistributed(MaxPooling1D(sent_maxlen))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    charOutput = Dropout(0.5)(char)

    return char_input, charOutput


def getResidualBiLSTM(sent_maxlen):
    '''
    Residual bilstm for word-level representation
    '''
    sess = tf.Session()
    K.set_session(sess)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    input_text = Input(shape=(sent_maxlen,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    word = Bidirectional(LSTM(units=512,
                              return_sequences=True,
                              recurrent_dropout=0.5,
                              dropout=0.5,
                              kernel_regularizer=regularizers.l2(0.001)))(embedding)
    word_ = Bidirectional(LSTM(units=512,
                               return_sequences=True,
                               recurrent_dropout=0.5,
                               dropout=0.5,
                               kernel_regularizer=regularizers.l2(0.001)))(word)
    word_representations = add([word, word_])  # residual connection

    return input_text, word_representations,


def build_bilstm_cnn_model(sent_maxlen, word_maxlen, char_vocab, dataset_type):
    """

    :param sent_maxlen:
    :param word_maxlen:
    :param char_vocab:
    :param dataset_type:
    :return:
    """

    char_vocab_size = len(char_vocab) + 1

    input_char, char_out = getCharCNN(sent_maxlen, word_maxlen, char_vocab_size)
    input_word, word_representations = getResidualBiLSTM(sent_maxlen)

    concat = merge([char_out, word_representations], mode='concat',
                   concat_axis=2)  # Residual and Highway connections are concatenated
    ##Fully-connected layer , here, the BiLSTM is acting as a decoder which will then predict probbblities over the
    #classes. for example, if you have 5 classes so it will
    ner_lstm = Bidirectional(LSTM(units=200,
                                  return_sequences=True,
                                  recurrent_dropout=0.3,
                                  dropout=0.3))(concat)
    #for example, if you have 5 classes so the Dense Net will assign 5 neurons to yield the softmax probablities over 5 classes.
    # [0.0232 0.3565 0.7644 0.8655 0.999]
    out = TimeDistributed(Dense(dataset_type, activation="softmax"))(ner_lstm)
    model = Model(inputs=[input_char, input_word],
                  outputs=out,
                  name='NER_Model')
    return model
