from builtins import range
import numpy as np
from cs231n.im2col import *
temp = {}

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    X   = np.reshape(x,(x.shape[0],-1))
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out  = X.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    X   = np.reshape(x,(x.shape[0],-1))
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    db = np.sum(dout,axis = 0)
    dw = np.matmul(dout.T,X)
    dx = np.matmul(dout,w.T)
    dw = dw.T
    dx = np.reshape(dx,x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
   
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################


    xc = x.copy()
    xc[xc<0] = 0
    out = xc
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.where(x > 0,dout, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    sample_mean = np.mean(x,axis = 0)
    sample_var  = np.var(x,axis =0)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    out= None
    cache = {}
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out = xhat * gamma + beta

        cache['x']      = x
        cache['xhat']   = xhat
        cache['gamma']   = gamma
        cache['beta']   = beta
        cache['running_mean'] = running_mean
        cache['running_var']  = running_var
        cache['eps'] = eps
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = (x - running_mean) / (np.sqrt(running_var + eps))
        out = out * gamma + beta
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    return batchnorm_backward_alt(dout, cache)

def batchnorm_backward_alt(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    N,D = dout.shape
    dx, dgamma, dbeta = None, None, None

    xhat    = cache['xhat']
    gamma    = cache['gamma']
    beta    = cache['beta']
    running_mean    = cache['running_mean']
    running_var    = cache['running_var']
    eps    = cache['eps']
    x      = cache['x']
    
    
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    dbeta = np.sum(dout,axis = 0)
    dgamma = np.sum(xhat*dout,axis = 0)
    dxhat = dout * gamma
    drv   = np.sum(dxhat * (x - running_mean) * -0.5 * np.power(running_var + eps,-1.5)     ,axis = 0)
    drm   = np.sum(dxhat * (-1.0/(np.sqrt(running_var + eps))),axis = 0) + drv * np.mean(-2.0 * (x - running_mean), axis = 0)
    dx    = dxhat * ((1.0/np.sqrt(running_var + eps)))   +  ( drv * (2.0/N) * (x - running_mean) )  +    (drm *1.0/N) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    sample_mean = np.mean(x,axis = 0)
    sample_var  = np.var(x,axis =0)
    xhat = (x - running_mean) / np.sqrt(running_var + eps)
    out = xhat * gamma + beta
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask =( np.random.rand(*x.shape) < p ) / p 
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        out = x * mask
        cache = (dropout_param,mask)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx
def conv_forward_naive(X, W, b, conv_param):
    padding ,  stride = conv_param['pad'] , conv_param['stride']
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = np.dot(W_col , X_col).T + b
    out = out.T
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding,X_col)

    return out, cache

##def conv_forward_naive(x, w, b, conv_param):
##    """
##    A naive implementation of the forward pass for a convolutional layer.
##
##    The input consists of N data points, each with C channels, height H and
##    width W. We convolve each input with F different filters, where each filter
##    spans all C channels and has height HH and width WW.
##
##    Input:
##    - x: Input data of shape (N, C, H, W)
##    - w: Filter weights of shape (F, C, HH, WW)
##    - b: Biases, of shape (F,)
##    - conv_param: A dictionary with the following keys:
##      - 'stride': The number of pixels between adjacent receptive fields in the
##        horizontal and vertical directions.
##      - 'pad': The number of pixels that will be used to zero-pad the input. 
##        
##
##    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
##    along the height and width axes of the input. Be careful not to modfiy the original
##    input x directly.
##
##    Returns a tuple of:
##    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
##      H' = 1 + (H + 2 * pad - HH) / stride
##      W' = 1 + (W + 2 * pad - WW) / stride
##    - cache: (x, w, b, conv_param)
##    """
##    
##    N , C ,H , W = x.shape
##    F , C,HH , WW = w.shape
##    pad ,  stride = conv_param['pad'] , conv_param['stride']
##    padded_x = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode = 'constant')
##  
##    ###########################################################################
##    # TODO: Implement the convolutional forward pass.                         #
##    # Hint: you can use the function np.pad for padding.                      #
##    ###########################################################################
##    Hout = 1 + (H + 2 * pad - HH) / stride
##    Wout = 1 + (W + 2 * pad - WW) / stride
##    out_shape = (N,F,Hout,Wout)
##    out = np.zeros(out_shape)
##    for f in range(F):
##        kernel = w[f]
##        bias   = b[f] 
##        for i in range(Hout):
##            hight  =  i * stride
##            for j in range(Wout):
##                width = j * stride
##                out[:,f,i,j] =  np.sum(padded_x[:,:,hight:hight+HH,width:width+WW] * kernel  ,axis = (1,2,3)) + bias
##    ###########################################################################
##    #                             END OF YOUR CODE                            #
##    ###########################################################################
##    cache = (x, w, b, conv_param)
##    return out, cache
##

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    #db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = np.dot(dout_reshaped , X_col.T)
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = np.dot(W_reshape.T , dout_reshaped)
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dX, dW, db


def max_pool_forward_naive(X, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    n , d , h , w  = X.shape
    pool_height , pool_width , stride  = pool_param['pool_height'] , pool_param['pool_width'], pool_param['stride']
    h_out = 1 + (h - pool_height) / stride
    w_out = 1 + (w - pool_width) / stride
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    X_reshaped = X.reshape(n * d, 1, h, w)

    # The result will be 4x9800
    # Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
    X_col = im2col_indices(X_reshaped, pool_height, pool_width, padding=0, stride=stride)

    # Next, at each possible patch location, i.e. at each column, we're taking the max index
    max_idx = np.argmax(X_col, axis=0)

    # Finally, we get all the max value at each column
    # The result will be 1x9800
    out = X_col[max_idx, range(max_idx.size)]

    # Reshape to the output size: 14x14x5x10
    out = out.reshape(h_out, w_out, n, d)

    # Transpose to get 5x10x14x14 output
    out = out.transpose(2, 3, 0, 1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (X, pool_param,X_col,max_idx,n , d , h , w)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    X, pool_param,X_col,max_idx ,n , d , h , w= cache
    pool_height , pool_width , stride  = pool_param['pool_height'] , pool_param['pool_width'], pool_param['stride']
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # X_col and max_idx are the intermediate variables from the forward propagation step

    # Suppose our output from forward propagation step is 5x10x14x14
    # We want to upscale that back to 5x10x28x28, as in the forward step

    # 4x9800, as in the forward step
    dX_col = np.zeros_like(X_col)

    # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
    # Transpose step is necessary to get the correct arrangement
    dout_flat = dout.transpose(2, 3, 0, 1).ravel()

    # Fill the maximum index of each column with the gradient

    # Essentially putting each of the 9800 grads
    # to one of the 4 row in 9800 locations, one at each column
    dX_col[max_idx, range(max_idx.size)] = dout_flat

    # We now have the stretched matrix of 4x9800, then undo it with col2im operation
    # dX would be 50x1x28x28
    dX = col2im_indices(dX_col, (n * d, 1, h, w), pool_height, pool_width, padding=0, stride=stride)

    # Reshape back to match the input dimension: 5x10x28x28
    dX = dX.reshape(X.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dX

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ##########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    ##########################################################################

    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    if mode == 'train':
        # Step 1 , calcul the average for each channel
        mu = (1. / (N * H * W) * np.sum(x, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        var = (1. / (N * H * W) * np.sum((x - mu)**2,
                                         axis=(0, 2, 3))).reshape(1, C, 1, 1)
        xhat = (x - mu) / (np.sqrt(eps + var))
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)

        running_mean = momentum * running_mean + \
            (1.0 - momentum) * np.squeeze(mu)
        running_var = momentum * running_var + \
            (1.0 - momentum) * np.squeeze(var)

        cache = (mu, var, x, xhat, gamma, beta, bn_param)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

    elif mode == 'test':
        mu = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)

        xhat = (x - mu) / (np.sqrt(eps + var))
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache

##def spatial_batchnorm_forward(x, gamma, beta, bn_param):
##    """
##    Computes the forward pass for spatial batch normalization.
##
##    Inputs:
##    - x: Input data of shape (N, C, H, W)
##    - gamma: Scale parameter, of shape (C,)
##    - beta: Shift parameter, of shape (C,)
##    - bn_param: Dictionary with the following keys:
##      - mode: 'train' or 'test'; required
##      - eps: Constant for numeric stability
##      - momentum: Constant for running mean / variance. momentum=0 means that
##        old information is discarded completely at every time step, while
##        momentum=1 means that new information is never incorporated. The
##        default of momentum=0.9 should work well in most situations.
##      - running_mean: Array of shape (D,) giving running mean of features
##      - running_var Array of shape (D,) giving running variance of features
##
##    Returns a tuple of:
##    - out: Output data, of shape (N, C, H, W)
##    - cache: Values needed for the backward pass
##    """
##    out, cache = None, None
##
##    ###########################################################################
##    # TODO: Implement the forward pass for spatial batch normalization.       #
##    #                                                                         #
##    # HINT: You can implement spatial batch normalization by calling the      #
##    # vanilla version of batch normalization you implemented above.           #
##    # Your implementation should be very short; ours is less than five lines. #
##    ###########################################################################
##    n , d , h , w  = x.shape
##    ###########################################################################
##    # TODO: Implement the max-pooling forward pass                            #
##    ###########################################################################
##    x_reshaped = x.reshape((n*d,-1))
##    b_repeated = np.tile(beta,n)
##    g_repeated = np.tile(gamma,n)
##    out,cache       = batchnorm_forward(x_reshaped.T, g_repeated, b_repeated, bn_param)
##    ###########################################################################
##    #                             END OF YOUR CODE                            #
##    ###########################################################################
##    out = np.reshape(out,x.shape)
##    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
