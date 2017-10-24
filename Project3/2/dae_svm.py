import tensorflow as tf
import sys
sys.path.insert(0, '/home/momi/Documents/599/Project3/utils/')

from autoencoder import DenoisingAutoencoder as DAE
import datasets

dae = DAE(model_name='dae_svm', pickle_name='svm', test_name='svm',
             n_components=256, main_dir='dae/', 
             enc_act_func='relu', dec_act_func='none', 
             loss_func='mean_squared', num_epochs=50, batch_size=100, 
             dataset='cifar10', xavier_init=1, opt='momentum', 
             learning_rate=0.001, momentum=0.5, corr_type='gaussian',
             corr_frac=0.3, verbose=1, seed=1)    

trX, trY, teX, teY = datasets.load_cifar10_dataset('../cifar-10-batches-py/',
                                                   mode='supervised')
val_dict = {}
dae.fit(trX, val_dict, teX, restore_previous_model=False) 
dae_svm_data = dae.transform(trX, name='train', save=False)

