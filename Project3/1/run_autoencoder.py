import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')

from autoencoder import DenoisingAutoencoder as DAE
import getdata

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('use_tf_flags', False, 'Parse arguments from CLI')
flags.DEFINE_string('model_name', 'first_test', 'Model name.')
flags.DEFINE_string('pickle_name', 'first_test_p', 'Pickle name.')
flags.DEFINE_string('test_name', 'first_test_t', 'Test case name.')
flags.DEFINE_string('dataset', 'cifar10', 'Which dataset to use. ["mnist", "cifar10"]')
flags.DEFINE_string('cifar_dir', '../cifar-10-batches-py/', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')


# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_components', 256, 'Number of hidden units in the dae.')
flags.DEFINE_string('corr_type', 'gaussian', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0.5, 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('main_dir', 'dae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('loss_func', 'mean_squared', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('weight_images', 0, 'Number of weight images to generate.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad","adam", "momentum"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_integer('num_epochs', 11, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')

assert FLAGS.dataset in ['mnist', 'cifar10']
assert FLAGS.enc_act_func in ['sigmoid', 'tanh', 'relu']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'relu', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'gaussian', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']


trX, teX = getdata.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
vlX = teX[:5000]  # Validation set is the first half of the test set

if len(sys.argv) == 1:
    print "Please enter an argument: \n [ncomp] - Test hidden layers" \
    "\n [lrs] - Test for learning rates" \
    "\n [bs] - Test for batch size" \
    "\n [act_fn] - Test for activation functions" \
    "\n [opt] - Test for optimizers" \
    "\n [momentum] - Test for momentum optimizer" \
    "\n [corr] - Test for corruption types" \
    "\n [ratio] - Test for corruption ratio" \
    "\n [loss] - Test for cost functions" 
    

if not FLAGS.use_tf_flags:    
    if len(sys.argv) > 1:
        val_dict ={}
        arg = sys.argv[1]
        # =============================================================================
        #       Testing hidden layers      
        # =============================================================================   
        if arg == 'ncomp':
            feed_list = [2**x for x in range(2,12)]    
            for i in feed_list:
                print "\n Evaluating for ncomp=" +str(i)
                t = 'hlayer=' + str(i)
                dae = DAE(model_name='hidden_layers', pickle_name=arg, test_name=t,
                         n_components=i, main_dir='hidden_layers/', 
                         enc_act_func='sigmoid', dec_act_func='sigmoid', 
                         loss_func='mean_squared', num_epochs=31, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=0.001, momentum=0.5, corr_type='gaussian',
                         corr_frac=0.6, verbose=1, seed=-1)
                dae.fit(trX, val_dict, teX, restore_previous_model=False)
                dae.reset()
        
        # =============================================================================
        #         Testing learning rates
        # =============================================================================        
        elif arg == 'lrs':
            feed_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]    
            for i in feed_list:
                print "\n Evaluating for lr=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func='tanh', dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=i, momentum=0.5, corr_type='gaussian',
                         corr_frac=0.3, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()
           
        # =============================================================================
        #         Testing batch sizes
        # =============================================================================        
        elif arg == 'bs':
            feed_list = [10, 25, 50, 64, 100, 200, 300, 400, 500]    
            for i in feed_list:
                print "\n Evaluating for bs=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func='tanh', dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=i, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=0.001, momentum=0.5, corr_type='gaussian',
                         corr_frac=0.3, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()
                
                                    
        # =============================================================================
        #         Testing activations
        # =============================================================================        
        elif arg == 'act_fn':
            feed_list = ['sigmoid', 'tanh', 'relu', 'None']
            for i in feed_list:
                print "\n Evaluating for act_fn=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func=i, dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=0.001, momentum=0.5, corr_type='gaussian',
                         corr_frac=0.3, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()
                
        # =============================================================================
        #         Testing optimizers
        # =============================================================================        
        elif arg == 'opt':
            feed_list = ['gradient_descent', 'ada_grad', 'momentum', 'adam']
            for i in feed_list:
                print "\n Evaluating for optimizer=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func='sigmoid', dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt=i, 
                         learning_rate=0.001, momentum=0.5, corr_type='gaussian',
                         corr_frac=0.3, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()

                  
        # =============================================================================
        #         Testing momentum
        # =============================================================================        
        elif arg == 'momentum':
            feed_list = [0.5,0.6,0.7,0.8,0.9]
            for i in feed_list:
                print "\n Evaluating for optimizer=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func='sigmoid', dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt='momentum', 
                         learning_rate=0.001, momentum=i, corr_type='gaussian',
                         corr_frac=0.3, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()

                  
        # =============================================================================
        #         Testing corruption types
        # =============================================================================        
        elif arg == 'corr':
            feed_list = ['masking', 'salt_and_pepper', 'gaussian','none']
            for i in feed_list:
                print "\n Evaluating for corruption=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name='corrupt', test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func='sigmoid', dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=0.001, momentum=0.5, corr_type=i,
                         corr_frac=0.3, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()
      
        # =============================================================================
        #         Testing corruption ratio
        # =============================================================================        
        elif arg == 'ratio':
            feed_list = [x/10.0 for x in range(1,11)]
            for i in feed_list:
                print "\n Evaluating for corruption=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=256, main_dir='hidden_layers/', 
                         enc_act_func='sigmoid', dec_act_func='none', 
                         loss_func='mean_squared', num_epochs=100, batch_size=12, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=0.001, momentum=0.5, corr_type='gaussian',
                         corr_frac=i, verbose=1, seed=-1)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()

    
        # =============================================================================
        #         Testing loss function
        # =============================================================================        
        elif arg == 'loss':
            feed_list = ['cross_entropy']#, 'mean_squared']
            for i in feed_list:
                print "\n Evaluating for cost=" +str(i)
                t = arg + '=' + str(i)
                dae = DAE(model_name=arg + '_model', pickle_name=arg, test_name=t,
                         n_components=1024, main_dir='hidden_layers/', 
                         enc_act_func='sigmoid', dec_act_func='sigmoid', 
                         loss_func=i, num_epochs=1000, batch_size=256, 
                         dataset='cifar10', xavier_init=1, opt='adam', 
                         learning_rate=0.0001, momentum=0.9, corr_type='gaussian',
                         corr_frac=0.2, verbose=1, seed=29)    
                dae.fit(trX, val_dict, teX, restore_previous_model=False) 
                dae.reset()
       

elif FLAGS.use_tf_flags: 

    dae = DAE(seed=FLAGS.seed, 
              model_name=FLAGS.model_name, 
              pickle_name=FLAGS.pickle_name,
              test_name=FLAGS.test_name,
              n_components=FLAGS.n_components,
              enc_act_func=FLAGS.enc_act_func,
              dec_act_func=FLAGS.dec_act_func,
              xavier_init=FLAGS.xavier_init,
              corr_type=FLAGS.corr_type,
              corr_frac=FLAGS.corr_frac,
              dataset=FLAGS.dataset,
              loss_func=FLAGS.loss_func,
              main_dir=FLAGS.main_dir,
              opt=FLAGS.opt,
              learning_rate=FLAGS.learning_rate,
              momentum=FLAGS.momentum,
              verbose=FLAGS.verbose,
              num_epochs=FLAGS.num_epochs,
              batch_size=FLAGS.batch_size)
    
    r = {}
    # Fit the model
    dae.fit(trX, r, teX, restore_previous_model=FLAGS.restore_previous_model)
#        dae.reset()
    # Encode the training data and store it
   # dae.transform(trX, name='train', save=False)
#    dae.transform(vlX, name='validation', save=FLAGS.encode_valid)
#    dae.transform(teX, name='test', save=FLAGS.encode_test)

    # save images
    #dae.get_weights_as_images(28, 28, max_images=FLAGS.weight_images)

