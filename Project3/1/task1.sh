#!/bin/bash/
nice -n 15 python run_autoencoder.py lrs
nice -n 15 python run_autoencoder.py bs
nice -n 15 python run_autoencoder.py act_fn
nice -n 15 python run_autoencoder.py opt
nice -n 15 python run_autoencoder.py momentum
nice -n 15 python run_autoencoder.py corr
nice -n 15 python run_autoencoder.py ratio
nice -n 15 python run_autoencoder.py loss
echo 'Done'

