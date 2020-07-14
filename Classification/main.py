import gc
import logging

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from numpy.random import seed

from algorithm.network import build_bilstm_cnn_model, network_model

from evaluation.eval_script import get_wnut_evaluation
from processed.Preprocess import start_build_sequences
from utilities.setting import (
    B,
    wnut_b,
    BASE_MODEL, )
from utilities.utilities import getLabels, save_predictions
import tensorflow as tf
from keras import backend as K

seed(7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class TextClassification:
    def __init__(self, architecture, batch_size, n_epochs, patience, lr_r):
        self.architecture = architecture
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.patience = patience
        self.lr_r = lr_r

    def run(self, filename=None, dataset_type=None, model_file=None, label_vocab=None):
        """
        Builds an NER model, predicts, saves prediction files, loads evaulation
        F1-scores from eval script (WNUT 2017 evaluation script cited in the paper)
        """

        logger.info("Preparing data initiated")
        (
            train_sent,
            dev_sent,
            test_sent,
            X_train,
            X_dev,
            X_test,
            x_c,
            xc_d,
            xc_t,
            y,
            y_d,
            y_t,
            char_lookup,
            sent_maxlen,
            word_maxlen,
        ) = start_build_sequences(vocabulary=wnut_b)
        y = y.reshape(y.shape[0], y.shape[1], 1)
        y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], 1)
        y_d = y_d.reshape(y_d.shape[0], y_d.shape[1], 1)

        model = network_model(sent_maxlen
                              , word_maxlen
                              , char_lookup
                              , dataset_type
                              , architecture=self.architecture,
                              )
        checkpointer = ModelCheckpoint(
            filepath="models/" + model_file + ".hdf5", verbose=1, save_best_only=True
        )
        earlystopper = EarlyStopping(
            monitor="val_loss", patience=self.patience, verbose=1
        )
        rms = RMSprop(lr=self.lr_r, rho=0.9, epsilon=None, decay=0.0)
        model.compile(
            optimizer=rms, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        model.summary()
        if self.architecture == BASE_MODEL:
            model.fit(
                [np.array(x_c), np.array(X_train)],
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,
                callbacks=[checkpointer, earlystopper],
                validation_data=([np.array(xc_d), np.array(X_dev)], y_d),
                shuffle=True,
            )
            predict = model.predict(
                [np.array(xc_t), np.array(X_test)],
                verbose=1,
                batch_size=self.batch_size,
            )
            self.get_prediction(
                X_test, y_t, predict, filename, label_vocab
            )
            get_wnut_evaluation(filename)


    def get_prediction(self, x, y, predict, filename, label_vocab):
        prediction = np.argmax(predict, axis=-1)
        prediction_final = np.array(prediction).tolist()
        predictions = getLabels(prediction_final, vocabulary=label_vocab)
        true = getLabels(y, vocabulary=label_vocab)
        save_predictions(filename, x, true, predictions)
        get_wnut_evaluation(filename)


if __name__ == "__main__":
    experiments = [
        BASE_MODEL,
    ]

    for id, exp in enumerate(experiments, start=0):
        print("Running %s model " % exp)
        ner = TextClassification(
            architecture=exp, batch_size=100, n_epochs=100, patience=10, lr_r=0.001
        )
        ner.run(
            filename="00" + str(id) + ".tsv",
            dataset_type=B,
            model_file="textual" + str(id) + "model",
            label_vocab=wnut_b,
        )

        print("Building results for".format(exp))

        print("/------------------------End Experiment------------------------------/")
