
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

def residual_block(x, channelIn, channelOut, scope):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        ####################################################################################################
        weight1 = tf.get_variable('weight1', shape = [1,1,channelIn,channelOut//2], dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.01))
        out = tf.contrib.layers.batch_norm(x, scope='bn1')
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, weight1, strides=[1, 1, 1, 1], padding='SAME')
        ####################################################################################################
        weight2 = tf.get_variable('weight2', shape = [3,3,channelOut//2,channelOut//2], dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.01))
        out = tf.contrib.layers.batch_norm(out, scope='bn2')
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, weight2, strides=[1, 1, 1, 1], padding='SAME')
        ####################################################################################################
        weight3 = tf.get_variable('weight3', shape = [1,1,channelOut//2,channelOut], dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev=0.01))
        out = tf.contrib.layers.batch_norm(out, scope='bn3')
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, weight3, strides=[1, 1, 1, 1], padding='SAME')
        ####################################################################################################
        weightUp = tf.get_variable('weightUp', shape = [1,1,channelIn,channelOut], dtype = tf.float32, 
                                   initializer = tf.truncated_normal_initializer(stddev=0.01))
        outUp = tf.nn.conv2d(x, weightUp, strides=[1, 1, 1, 1], padding='SAME')
        ####################################################################################################
        out = tf.add(out,outUp)
        return out
        
def hourglass_block(x, channelIn, channelOut, scope):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        out = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        out = residual_block(out, channelIn, 64, scope='res1')
        out = residual_block(out, 64, 64, scope='res2')
        out = residual_block(out, 64, 64, scope='res3')
        out = residual_block(out, 64, channelOut, scope='res4')
        out = residual_block(out, channelOut, channelOut, scope='res5')
        out = tf.image.resize_nearest_neighbor(out, [tf.shape(x)[1],tf.shape(x)[2]])
        ####################################################################################################
        outUp = residual_block(x, channelIn, 64, scope='resUp1')
        outUp = residual_block(outUp, 64, 64, scope='resUp2')
        outUp = residual_block(outUp, 64, channelOut, scope='resUp3')
        ####################################################################################################
        out = tf.add(out,outUp)
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
