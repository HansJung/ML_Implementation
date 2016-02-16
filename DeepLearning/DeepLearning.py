__author__ = 'jeong-yonghan'

import cPickle, gzip
import numpy as np
import theano
import theano.tensor as T

# Download the large size dataset
f = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# print valid_set[0][0].shape


def shared_dataset(data_xy):
    """
    Function that loads the dataset into shared variables

    This is a exchanger between the set of minibatch data and GPU memory
    """

    data_x,data_y = data_xy # Dividing by dataset / label
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500

# # For example, if we want to access the third minibatch of the training set
# Access_idx = 3
# data  = train_set_x[Access_idx-1 * batch_size: Access_idx * batch_size] # 0 to 1 // 1 to 2 // 2 to 3
# label = train_set_y[Access_idx-1 * batch_size: Access_idx * batch_size]

# Loss function
## zero-one loss
def zero_one_loss(y,p_y_given_x):
    return T.sum(T.neq(T.argmax(p_y_given_x) , y))

## NLL loss
def NLL(y,p_y_given_x):
    return -T.sum(  T.log(p_y_given_x)[T.arange(y.shape[0],y)]  )


