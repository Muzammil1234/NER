from random import seed
from nltk.corpus.reader.conll import ConllCorpusReader
from nltk.tokenize.treebank import TreebankWordDetokenizer

from utilities.setting import DEV, TEST, TRAIN
from utilities.utilities import create_lookup, create_sequences

seed(7)


def conllReader(corpus):
    '''
    Data reader for CoNLL format data
    '''
    root = "data/"
    sentences = []

    ccorpus = ConllCorpusReader(root, ".conll", ('words', 'pos', 'tree'))

    raw = ccorpus.sents(corpus)

    for sent in raw:
        sentences.append([TreebankWordDetokenizer().detokenize(sent)])

    tagged = ccorpus.tagged_sents(corpus)
    print(tagged)


    return tagged, sentences


def build_lookups(train_name, dev_name, test_name, voc):
    '''
    Wrapper function to builds lookups for characters and labels. Also, computes the sentence and word max lengths.
    '''
    sentences = []
    sentence = []
    sent_maxlen = 0
    word_maxlen = 0
    root = "data/"

    for fname in (train_name, dev_name, test_name):
        with open((root + fname), 'r', encoding='utf8') as file:
            for line in file:
                line = line.rstrip()
                if line == '':
                    sent_maxlen = max(sent_maxlen, len(sentence))
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(line.split('\t'))
                    word_maxlen = max(word_maxlen, len(str(line.split()[0])))

    sentences.append(sentence)
    num_sentence = len(sentences)

    char_lookup, label_lookup = create_lookup(sentences, voc)

    # considring a higher valued max sent_length

    sent_maxlen= 100
    return [sentences, sent_maxlen, word_maxlen, num_sentence, char_lookup, label_lookup]





def flatten(list):
    '''
    Helper function that flattens 2D lists.
    '''
    return [i for sublist in list for i in sublist]


def sequence_helper(x_in, sent_maxlen, casing=False):
    '''
    Helper function for word sequences (text data sepcific)
    :param x_in:
    :param sent_maxlen:
    :return: Word sequences
    '''

    new_X = []
    for seq in x_in:
        new_seq = []
        for i in range(sent_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append('__pad__')
        new_X.append(new_seq)
    return new_X



def start_build_sequences(vocabulary):
    '''
    Sequence builder for text data specific
    :param vocabulary: Label vocabulary specific for the datasets
    :return: Sequences
    '''
    sentences, sent_maxlen, word_maxlen, \
    num_sentence, char_lookup, label_lookup = build_lookups(TRAIN, DEV, TEST, vocabulary)
    train_sent, train_dt_sent = conllReader(TRAIN)
    dev_sent, dev_dt_sent = conllReader(DEV)
    test_sent, test_dt_sent = conllReader(TEST)

    # logger.info('Setting up input sequences')
    x, y, x_c, addCharTrain = create_sequences(train_sent, char_lookup, label_lookup, word_maxlen, sent_maxlen)
    x_t, y_t, xc_t, addCharTest = create_sequences(test_sent, char_lookup, label_lookup, word_maxlen, sent_maxlen)
    x_d, y_d, xc_d, addCharDev = create_sequences(dev_sent, char_lookup, label_lookup, word_maxlen, sent_maxlen)
    X_train = sequence_helper(x, sent_maxlen)
    X_test = sequence_helper(x_t, sent_maxlen)
    X_dev = sequence_helper(x_d, sent_maxlen)

    return [train_dt_sent, dev_dt_sent, test_dt_sent, X_train, X_dev, X_test, x_c, xc_d, xc_t, y, y_d, y_t, char_lookup,
            sent_maxlen, word_maxlen]

