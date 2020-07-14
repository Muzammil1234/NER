from enum import Enum

import h5py

######################################################
#                     Architectures                  #
######################################################
BASE_MODEL = 'base'
EXTENDED_BASE_MODEL = 'extended_base'


######################################################
#                     Important constants            #
######################################################

B = 13

UNK_TOKEN= '<UNK>'
PAD_TOKEN= '<PAD>'

######################################################
#                     Data Set Paths                 #
######################################################
# WNUT 2017

TRAIN = "training.conll"

DEV = "dev.conll"

TEST = "test.conll"


######################################################
#                     Label vocabularies             #
######################################################


wnut_b = { 'B-corporation':12,
    'B-creative-work':11,
    'B-group':10,
    'B-location':9,
    'B-person':8,
    'B-product':7,
    'I-corporation':6,
    'I-creative-work':5,
    'I-group':4,
    'I-location':3,
    'I-person':2,
    'I-product':1,
    'O':0}

label_b = ["person", "location", "creative-work", "corporation", "product", "group"]

