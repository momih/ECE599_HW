#!/bin/bash/
nice -n 15 python ConvAE.py
nice -n 15 python cnn_ae_weights.py init
nice -n 15 python cnn_ae_weights.py none 

