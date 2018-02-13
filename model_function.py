
import tensorflow as tf

def max_pool(x, k=2, s=2):
    return tf.nn.max_pool( x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')
	
def avg_pool(x, k=2, s=2):
    return tf.nn.avg_pool( x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')
    
def flatten(x):
    return tf.contrib.layers.flatten(x)
        
def linear(x, channelIn, channelOut, scope, addBias=False, activated=False):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        weight = tf.get_variable('weight', shape = [channelIn,channelOut], dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.01))
        out = tf.matmul(x, weight)
        #====================================================================================================
        if addBias :
            bias = tf.get_variable('bias', shape = [channelOut], dtype = tf.float32,
                                   initializer = tf.truncated_normal_initializer(stddev=0.01))
            out = tf.nn.bias_add(out, bias)
        #====================================================================================================
        if activated :
            out = tf.nn.leaky_relu(out)
        return out
	
def conv2d(x, channelIn, channelOut, scope, k=3, s=1, padding='SAME',
           addBias=False, batchNorm=False, activated=False):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        weight = tf.get_variable('weight', shape = [k,k,channelIn,channelOut], dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.01))
        out = tf.nn.conv2d(x, weight, strides=[1, s, s, 1], padding=padding)
        #====================================================================================================
        if addBias :
            bias = tf.get_variable('bias', shape = [channelOut], dtype = tf.float32,
                                   initializer = tf.truncated_normal_initializer(stddev=0.01))
            out = tf.nn.bias_add(out, bias)
        #====================================================================================================
        if batchNorm :
            out = tf.contrib.layers.batch_norm(out, scope='bn')
        #====================================================================================================
        if activated :
            out = tf.nn.leaky_relu(out)
        return out
    
def deconv2d(x, channelIn, channelOut, scope, k=3, s=1, padding='SAME',
             addBias=False, batchNorm=False, activated=False):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        weight = tf.get_variable('weight', shape = [k,k,channelOut,channelIn], dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.01))
        x_shape = tf.shape(x)
        w_shape = tf.shape(weight)
        outputShape = tf.stack([x_shape[0], x_shape[1]*s, x_shape[2]*s, w_shape[2]])
        out = tf.nn.conv2d_transpose(x, weight, outputShape, strides=[1, s, s, 1], padding=padding)
        #====================================================================================================
        if addBias :
            bias = tf.get_variable('bias', shape = [channelOut], dtype = tf.float32,
                                   initializer = tf.truncated_normal_initializer(stddev=0.01))
            out = tf.nn.bias_add(out, bias)
        #====================================================================================================
        if batchNorm :
            out = tf.contrib.layers.batch_norm(out, scope='bn')
        #====================================================================================================
        if activated :
            out = tf.nn.leaky_relu(out)
        return out
        
def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3) 
    
def crop(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    return tf.slice(x1, offsets, size)
    