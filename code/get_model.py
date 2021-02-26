import keras
from keras.models import Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten,Reshape,concatenate,LSTM,Bidirectional, Average
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D,AveragePooling2D
from keras.layers import Lambda, dot
import tensorflow as tf
#import string
from keras.utils import multi_gpu_model
from keras.utils import plot_model
import numpy as np
import os
from keras import regularizers
from keras import initializers
import tensorflow

def attention_3d_block(hidden_states,out_shape=128,name='pro_'):
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name=name+'attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name=name+'last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name=name+'attention_score')
    attention_weights = Activation('softmax', name=name+'attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name=name+'context_vector')
    pre_activation = concatenate([context_vector, h_t], name=name+'attention_output')
    attention_vector = Dense(out_shape, use_bias=False, activation='tanh', name=name+'attention_vector')(pre_activation)
    return attention_vector

def conv2d_bn(x, nb_filter, num_row, num_col,name,
              padding='same', strides=(1, 1), use_bias=False):
    x = Conv2D(nb_filter, (num_row, num_col),
                      name=name,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    x = Activation('relu')(x)
    return x

def block_inception(input,filters_1x1, filters_3x3_reduce, filters_3x3,
                      filters_5x5_reduce, filters_5x5, filters_pool_proj,layer_name):
    branch_0 = conv2d_bn(input, filters_1x1, 1, 1,name=layer_name+'_branch_0')
    branch_1 = conv2d_bn(input, filters_3x3_reduce, 1, 1,name=layer_name+'_branch_1_3x3_reduce')
    branch_1 = conv2d_bn(branch_1, filters_3x3, 3, 3,name=layer_name+'_branch_1_3x3')
    branch_2 = conv2d_bn(input, filters_5x5_reduce, 1, 1,name=layer_name+'_branch_2_5x5_reduce')
    branch_2 = conv2d_bn(branch_2, filters_5x5, 3, 3,name=layer_name+'_branch_2_3x3_0')
    branch_2 = conv2d_bn(branch_2, filters_5x5, 3, 3,name=layer_name+'_branch_2_3x3_1')
    branch_3 = MaxPooling2D((3,3), strides=(1,1), padding='same',name=layer_name+'_branch_3_maxpooling')(input)#AveragePooling2D
    branch_3 = conv2d_bn(branch_3, filters_pool_proj, 1, 1,name=layer_name+'_branch_3_pool_proj')
    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1,name=layer_name+'_concat')
    return x

def block_inception_b(input,filters_1x1, filters_5x5_reduce, filters_5x5,
                      filters_7x7_reduce, filters_1x7,filters_7x1,filters_pool_proj,layer_name):
    branch_0 = conv2d_bn(input, filters_1x1, 1, 1,name=layer_name+'_branch_0')

    branch_1 = conv2d_bn(input,filters_7x7_reduce, 1, 1,name=layer_name+'_branch_1_7x7_reduce')
    branch_1 = conv2d_bn(branch_1,filters_1x7, 1, 7,name=layer_name+'_branch_1_7x7_0')
    branch_1 = conv2d_bn(branch_1,filters_7x1, 7, 1,name=layer_name+'_branch_1_7x7_1')

    branch_2 = conv2d_bn(input, filters_5x5_reduce, 1, 1,name=layer_name+'_branch_2_5x5_reduce')
    branch_2 = conv2d_bn(branch_2, filters_5x5, 3, 3,name=layer_name+'_branch_2_3x3_0')
    branch_2 = conv2d_bn(branch_2, filters_5x5, 3, 3,name=layer_name+'_branch_2_3x3_1')

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, filters_pool_proj, 1, 1,name=layer_name+'_branch_3_pool_proj')

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)#branch_2,
    return x

def simple_block(input,nb_filter,num_row,num_col,layer_name):
    input = Conv2D(nb_filter, (num_row, num_col), padding='same', activation='relu', name=layer_name+'_conv0')(input)
    input = Conv2D(nb_filter, (num_row, num_col), padding='same', activation='relu', name=layer_name+'_conv1')(input)
    return input

def get_model_classification(save_dir,alpha,
                                          pro_branch_switch1='',pro_branch_switch2='',
                                          pro_branch_switch3='',pro_add_attention=False,
                                          comp_branch_switch1='',comp_branch_switch2='',
                                          comp_branch_switch3='',comp_add_attention=False,
               ):
    ###MODEL
    ##input
    protein_input = Input(shape=(1200, 20, 1), name='protein_input')
    comp_input = Input(shape=(200, 67, 1), name='comp_input')
    ##protein branch
    # layer1
    with tf.device('/gpu:0'):
        if pro_branch_switch1 == 'inception_block':
            pro_layer1 = block_inception(protein_input, filters_1x1=8, filters_3x3_reduce=1, filters_3x3=32,
                                         filters_5x5_reduce=1, filters_5x5=32, filters_pool_proj=16,layer_name='pro_layer1')
        else:
            pro_layer1 = simple_block(protein_input, nb_filter=32, num_row=3, num_col=3, layer_name='pro_layer1')
        pro_layer1 = MaxPooling2D(pool_size=(3, 3),padding='same', name='pro_layer1_poll')(pro_layer1)
        # layer2
        if pro_branch_switch2=='inception_block':
            pro_layer2 = block_inception(pro_layer1, filters_1x1=16, filters_3x3_reduce=16, filters_3x3=64,
                                         filters_5x5_reduce=16, filters_5x5=64, filters_pool_proj=32,layer_name='pro_layer2')
        else:
            pro_layer2=simple_block(pro_layer1,nb_filter=64,num_row=3,num_col=3,layer_name='pro_layer2')
        pro_layer2 = MaxPooling2D(pool_size=(3, 3), padding='same',name='pro_layer2_poll')(pro_layer2)
            # layer3
        if pro_branch_switch3=='inception_block':
            pro_layer3 = block_inception(pro_layer2, filters_1x1=32, filters_3x3_reduce=64, filters_3x3=128,
                                         filters_5x5_reduce=64, filters_5x5=128, filters_pool_proj=64,layer_name='pro_layer3')
        elif pro_branch_switch3=='inception_block_b':
            pro_layer3 = block_inception_b(pro_layer2, filters_1x1=32, filters_5x5_reduce=64, filters_5x5=128,
                                         filters_7x7_reduce=64, filters_1x7=128,filters_7x1=128, filters_pool_proj=64,layer_name='pro_layer3')
        else:
            pro_layer3 = simple_block(pro_layer2, nb_filter=128, num_row=3, num_col=3, layer_name='pro_layer3')
        pro_layer3 = MaxPooling2D(pool_size=(3, 3), padding='same',name='pro_layer3_pool')(pro_layer3)
        # layer4
        if pro_add_attention:
            h_t = Lambda(tf.reshape,output_shape=[45,352,], arguments={'shape': [-1, 45, 352]}, name='pro_convert_to_timestep')(pro_layer3)
            pro_layer_tran_result = attention_3d_block(h_t,1024,'pro_')#batch*1024
        else:
            pro_layer_tran_result = Flatten(name='pro_layer4_flatten')(pro_layer3)
            pro_layer_tran_result = Dense(1024, activation='relu', name='pro_layer5_den')(pro_layer_tran_result)
        pro_layer_tran_result = Dropout(alpha, name='pro_drop1')(pro_layer_tran_result)
    ##compound branch
    # layer1
    with tf.device('/gpu:1'):
        if comp_branch_switch1=='inception_block':
            comp_layer1 = block_inception(comp_input, filters_1x1=8, filters_3x3_reduce=1, filters_3x3=16,
                                         filters_5x5_reduce=1, filters_5x5=16, filters_pool_proj=16,
                                         layer_name='comp_layer1')
        else:
            comp_layer1 = simple_block(comp_input, 32, 3, 3, 'comp_layer1')
        comp_layer1 = MaxPooling2D(pool_size=(2, 2),padding='same', name='comp_layer1_poll')(comp_layer1)
        # layer2
        if comp_branch_switch2=='inception_block':
            comp_layer2 = block_inception(comp_layer1, filters_1x1=16, filters_3x3_reduce=16, filters_3x3=64,
                                          filters_5x5_reduce=16, filters_5x5=64, filters_pool_proj=32,layer_name='comp_layer2')
        else:
            comp_layer2=simple_block(comp_layer1,64,3,3,'comp_layer2')
        comp_layer2 = MaxPooling2D(pool_size=(2, 2), padding='same',name='comp_layer2_poll')(comp_layer2)
        # layer3
        if comp_branch_switch3=='inception_block':
            comp_layer3 = block_inception(comp_layer2, filters_1x1=32, filters_3x3_reduce=32, filters_3x3=128,
                                          filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=32,layer_name='comp_layer3')
        elif comp_branch_switch3=='inception_block_b':
            comp_layer3 = block_inception_b(comp_layer2, filters_1x1=32, filters_5x5_reduce=32, filters_5x5=128,
                                           filters_7x7_reduce=32, filters_1x7=128, filters_7x1=128,
                                           filters_pool_proj=32, layer_name='comp_layer3')
        else:
            comp_layer3=simple_block(comp_layer2,128,3,3,'comp_layer3')
        comp_layer3 = MaxPooling2D(pool_size=(2, 2), padding='same',name='comp_layer3_pool')(comp_layer3)
        # layer4
        if comp_add_attention:
            h_t = Lambda(tf.reshape,output_shape=[25*8,320,], arguments={'shape': [-1, 25*8, 320]}, name='comp_convert_to_timestep')(comp_layer3)
            comp_layer_tran_result = attention_3d_block(h_t,1024,'comp_')#batch*1024
        else:
            comp_layer_tran_result = Flatten(name='comp_layer4_flatten')(comp_layer3)
            comp_layer_tran_result = Dense(640, activation='relu', name='comp_layer5_den')(comp_layer_tran_result)
        # layer5
        comp_layer_tran_result = Dropout(alpha, name='comp_drop1')(comp_layer_tran_result)
    with tf.device('/gpu:2'):
        pro_com = keras.layers.concatenate([pro_layer_tran_result, comp_layer_tran_result])
        # We stack a deep densely-connected network on top
        fc_pro_com = Dense(512, activation='relu', name='den1')(pro_com)
        fc_pro_com = Dropout(alpha, name='drop1')(fc_pro_com)
        dense1 = []
        FC1 = Dense(64, activation='relu')
        for p in np.linspace(0.1,0.5, 5):
            x = Dropout(p)(fc_pro_com)
            x = FC1(x)
            x = Dense(1,activation='sigmoid')(x)
            dense1.append(x)
        class_out = Average()(dense1)
    classification_model = Model([protein_input, comp_input],class_out)
    plot_model(classification_model, to_file=save_dir + '/model_with_classification.png', show_shapes=True)
    return classification_model

def get_model_regression(save_dir,alpha,
                                          pro_branch_switch1='',pro_branch_switch2='',
                                          pro_branch_switch3='',pro_add_attention=False,
                                          comp_branch_switch1='',comp_branch_switch2='',
                                          comp_branch_switch3='',comp_add_attention=False,
               ):
    ###MODEL
    ##input
    protein_input = Input(shape=(1200, 20, 1), name='protein_input')
    comp_input = Input(shape=(200, 67, 1), name='comp_input')
    ##protein branch
    # layer1
    with tf.device('/gpu:0'):
        if pro_branch_switch1 == 'inception_block':
            pro_layer1 = block_inception(protein_input, filters_1x1=8, filters_3x3_reduce=1, filters_3x3=32,
                                         filters_5x5_reduce=1, filters_5x5=32, filters_pool_proj=16,layer_name='pro_layer1')
        else:
            pro_layer1 = simple_block(protein_input, nb_filter=32, num_row=3, num_col=3, layer_name='pro_layer1')
        pro_layer1 = MaxPooling2D(pool_size=(3, 3),padding='same', name='pro_layer1_poll')(pro_layer1)
        # layer2
        if pro_branch_switch2=='inception_block':
            pro_layer2 = block_inception(pro_layer1, filters_1x1=16, filters_3x3_reduce=16, filters_3x3=64,
                                         filters_5x5_reduce=16, filters_5x5=64, filters_pool_proj=32,layer_name='pro_layer2')
        else:
            pro_layer2=simple_block(pro_layer1,nb_filter=64,num_row=3,num_col=3,layer_name='pro_layer2')
        pro_layer2 = MaxPooling2D(pool_size=(3, 3), padding='same',name='pro_layer2_poll')(pro_layer2)
            # layer3
        if pro_branch_switch3=='inception_block':
            pro_layer3 = block_inception(pro_layer2, filters_1x1=32, filters_3x3_reduce=64, filters_3x3=128,
                                         filters_5x5_reduce=64, filters_5x5=128, filters_pool_proj=64,layer_name='pro_layer3')
        elif pro_branch_switch3=='inception_block_b':
            pro_layer3 = block_inception_b(pro_layer2, filters_1x1=32, filters_5x5_reduce=64, filters_5x5=128,
                                         filters_7x7_reduce=64, filters_1x7=128,filters_7x1=128, filters_pool_proj=64,layer_name='pro_layer3')
        else:
            pro_layer3 = simple_block(pro_layer2, nb_filter=128, num_row=3, num_col=3, layer_name='pro_layer3')
        pro_layer3 = MaxPooling2D(pool_size=(3, 3), padding='same',name='pro_layer3_pool')(pro_layer3)
        # layer4
        if pro_add_attention:
            h_t = Lambda(tf.reshape,output_shape=[45,352,], arguments={'shape': [-1, 45, 352]}, name='pro_convert_to_timestep')(pro_layer3)#0000000000128
            # h_t = Bidirectional(LSTM(100, return_sequences=True),input_shape=(45,128))(h_t) #batch*tmiestep*100
            pro_layer_tran_result = attention_3d_block(h_t,1024,'pro_')#batch*1024
        else:
            pro_layer_tran_result = Flatten(name='pro_layer4_flatten')(pro_layer3)
            pro_layer_tran_result = Dense(1024, activation='relu', name='pro_layer5_den')(pro_layer_tran_result)
        pro_layer_tran_result = Dropout(alpha, name='pro_drop1')(pro_layer_tran_result)
    ##compound branch
    # layer1
    with tf.device('/gpu:1'):
        if comp_branch_switch1=='inception_block':
            comp_layer1 = block_inception(comp_input, filters_1x1=8, filters_3x3_reduce=1, filters_3x3=16,
                                         filters_5x5_reduce=1, filters_5x5=16, filters_pool_proj=16,
                                         layer_name='comp_layer1')
        else:
            comp_layer1 = simple_block(comp_input, 32, 3, 3, 'comp_layer1')
        comp_layer1 = MaxPooling2D(pool_size=(2, 2),padding='same', name='comp_layer1_poll')(comp_layer1)
        # layer2
        if comp_branch_switch2=='inception_block':
            comp_layer2 = block_inception(comp_layer1, filters_1x1=16, filters_3x3_reduce=16, filters_3x3=64,
                                          filters_5x5_reduce=16, filters_5x5=64, filters_pool_proj=32,layer_name='comp_layer2')
        else:
            comp_layer2=simple_block(comp_layer1,64,3,3,'comp_layer2')
        comp_layer2 = MaxPooling2D(pool_size=(2, 2), padding='same',name='comp_layer2_poll')(comp_layer2)
        # layer3
        if comp_branch_switch3=='inception_block':
            comp_layer3 = block_inception(comp_layer2, filters_1x1=32, filters_3x3_reduce=32, filters_3x3=128,
                                          filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=32,layer_name='comp_layer3')
        elif comp_branch_switch3=='inception_block_b':
            comp_layer3 = block_inception_b(comp_layer2, filters_1x1=32, filters_5x5_reduce=32, filters_5x5=128,
                                           filters_7x7_reduce=32, filters_1x7=128, filters_7x1=128,
                                           filters_pool_proj=32, layer_name='comp_layer3')
        else:
            comp_layer3=simple_block(comp_layer2,128,3,3,'comp_layer3')
        comp_layer3 = MaxPooling2D(pool_size=(2, 2), padding='same',name='comp_layer3_pool')(comp_layer3)
        # layer4
        if comp_add_attention:
            h_t = Lambda(tf.reshape,output_shape=[25*8,320,], arguments={'shape': [-1, 25*8, 320]}, name='comp_convert_to_timestep')(comp_layer3)#@@@@@@@@@128
            # h_t = Bidirectional(LSTM(100, return_sequences=True),input_shape=(25*8,128))(h_t) #batch*tmiestep*100
            comp_layer_tran_result = attention_3d_block(h_t,1024,'comp_')#batch*1024
        else:
            comp_layer_tran_result = Flatten(name='comp_layer4_flatten')(comp_layer3)
            comp_layer_tran_result = Dense(640, activation='relu', name='comp_layer5_den')(comp_layer_tran_result)
        # layer5
        # comp_layer_tran_result = Dense(256, activation='relu', name='comp_layer4_den')(comp_layer_tran_result)
        comp_layer_tran_result = Dropout(alpha, name='comp_drop1')(comp_layer_tran_result)
    with tf.device('/gpu:2'):
        pro_com = keras.layers.concatenate([pro_layer_tran_result, comp_layer_tran_result])
        # We stack a deep densely-connected network on top
        fc_pro_com = Dense(512, activation='relu', name='den1')(pro_com)
        fc_pro_com = Dropout(alpha, name='drop1')(fc_pro_com)
        dense1 = []
        FC1 = Dense(64, activation='relu')
        for p in np.linspace(0.1,0.5, 5):
            x = Dropout(p)(fc_pro_com)
            x = FC1(x)
            x = Dense(1)(x)
            dense1.append(x)
        class_out = Average()(dense1)
    regression_model = Model([protein_input, comp_input],class_out)
    plot_model(classification_model, to_file=save_dir + '/model_with_regression.png', show_shapes=True)
    return regression_model

def get_model_multi(save_dir,alpha,
                                          pro_branch_switch1='',pro_branch_switch2='',
                                          pro_branch_switch3='',pro_add_attention=False,
                                          comp_branch_switch1='',comp_branch_switch2='',
                                          comp_branch_switch3='',comp_add_attention=False,
               ):
    ###MODEL
    ##input
    protein_input = Input(shape=(1200, 20, 1), name='protein_input')
    comp_input = Input(shape=(200, 67, 1), name='comp_input')
    with tf.device('/gpu:0'):
        ##protein branch
        # layer1
        if pro_branch_switch1 == 'inception_block':
            pro_layer1 = block_inception(protein_input, filters_1x1=8, filters_3x3_reduce=1, filters_3x3=32,
                                         filters_5x5_reduce=1, filters_5x5=32, filters_pool_proj=16,layer_name='pro_layer1')
        else:
            pro_layer1 = simple_block(protein_input, nb_filter=32, num_row=3, num_col=3, layer_name='pro_layer1')
        pro_layer1 = MaxPooling2D(pool_size=(3, 3),padding='same', name='pro_layer1_poll')(pro_layer1)
        # layer2
        if pro_branch_switch2=='inception_block':
            pro_layer2 = block_inception(pro_layer1, filters_1x1=16, filters_3x3_reduce=16, filters_3x3=64,
                                         filters_5x5_reduce=16, filters_5x5=64, filters_pool_proj=32,layer_name='pro_layer2')
        else:
            pro_layer2=simple_block(pro_layer1,nb_filter=64,num_row=3,num_col=3,layer_name='pro_layer2')
        pro_layer2 = MaxPooling2D(pool_size=(3, 3), padding='same',name='pro_layer2_poll')(pro_layer2)
        # layer3
        if pro_branch_switch3=='inception_block':
            pro_layer3 = block_inception(pro_layer2, filters_1x1=32, filters_3x3_reduce=64, filters_3x3=128,
                                         filters_5x5_reduce=64, filters_5x5=128, filters_pool_proj=64,layer_name='pro_layer3')
        elif pro_branch_switch3=='inception_block_b':
            pro_layer3 = block_inception_b(pro_layer2, filters_1x1=32, filters_5x5_reduce=64, filters_5x5=128,
                                         filters_7x7_reduce=64, filters_1x7=128,filters_7x1=128, filters_pool_proj=64,layer_name='pro_layer3')
        else:
            pro_layer3 = simple_block(pro_layer2, nb_filter=128, num_row=3, num_col=3, layer_name='pro_layer3')
        pro_layer3 = MaxPooling2D(pool_size=(3, 3), padding='same',name='pro_layer3_pool')(pro_layer3)
        # layer4
        if pro_add_attention:
            h_t = Lambda(tf.reshape,output_shape=[45,352,], arguments={'shape': [-1, 45, 352]}, name='pro_convert_to_timestep')(pro_layer3)
            pro_layer_tran_result = attention_3d_block(h_t,1024,'pro_')#batch*1024
        else:
            pro_layer_tran_result = Flatten(name='pro_layer4_flatten')(pro_layer3)
            pro_layer_tran_result = Dense(1024, activation='relu', name='pro_layer5_den')(pro_layer_tran_result)
        pro_layer_tran_result = Dropout(alpha, name='pro_drop1')(pro_layer_tran_result)
    with tf.device('/gpu:1'):
        ##compound branch
        # layer1
        if comp_branch_switch1=='inception_block':
            comp_layer1 = block_inception(comp_input, filters_1x1=8, filters_3x3_reduce=1, filters_3x3=16,
                                         filters_5x5_reduce=1, filters_5x5=16, filters_pool_proj=16,
                                         layer_name='comp_layer1')
        else:
            comp_layer1 = simple_block(comp_input, 32, 3, 3, 'comp_layer1')
        comp_layer1 = MaxPooling2D(pool_size=(2, 2),padding='same', name='comp_layer1_poll')(comp_layer1)
        # layer2
        if comp_branch_switch2=='inception_block':
            comp_layer2 = block_inception(comp_layer1, filters_1x1=16, filters_3x3_reduce=16, filters_3x3=64,
                                          filters_5x5_reduce=16, filters_5x5=64, filters_pool_proj=32,layer_name='comp_layer2')
        else:
            comp_layer2=simple_block(comp_layer1,64,3,3,'comp_layer2')
        comp_layer2 = MaxPooling2D(pool_size=(2, 2), padding='same',name='comp_layer2_poll')(comp_layer2)
        # layer3
        if comp_branch_switch3=='inception_block':
            comp_layer3 = block_inception(comp_layer2, filters_1x1=32, filters_3x3_reduce=32, filters_3x3=128,
                                          filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=32,layer_name='comp_layer3')
        elif comp_branch_switch3=='inception_block_b':
            comp_layer3 = block_inception_b(comp_layer2, filters_1x1=32, filters_5x5_reduce=32, filters_5x5=128,
                                           filters_7x7_reduce=32, filters_1x7=128, filters_7x1=128,
                                           filters_pool_proj=32, layer_name='comp_layer3')
        else:
            comp_layer3=simple_block(comp_layer2,128,3,3,'comp_layer3')
        comp_layer3 = MaxPooling2D(pool_size=(2, 2), padding='same',name='comp_layer3_pool')(comp_layer3)
        # layer4
        if comp_add_attention:
            h_t = Lambda(tf.reshape,output_shape=[25*8,320,], arguments={'shape': [-1, 25*8, 320]}, name='comp_convert_to_timestep')(comp_layer3)
            comp_layer_tran_result = attention_3d_block(h_t,1024,'comp_')#batch*1024
        else:
            comp_layer_tran_result = Flatten(name='comp_layer4_flatten')(comp_layer3)
            comp_layer_tran_result = Dense(640, activation='relu', name='comp_layer5_den')(comp_layer_tran_result)
        # layer5
        comp_layer_tran_result = Dropout(alpha, name='comp_drop1')(comp_layer_tran_result)
    with tf.device('/gpu:2'):
        pro_com = keras.layers.concatenate([pro_layer_tran_result, comp_layer_tran_result])
        # We stack a deep densely-connected network on top
        fc_pro_com = Dense(512, activation='relu', name='den1')(pro_com)
        fc_pro_com = Dropout(alpha, name='drop1')(fc_pro_com)

        # classification task
        dense1 = []
        FC1 = Dense(64, activation='relu')
        for p in np.linspace(0.1,0.5, 5):
            x = Dropout(p)(fc_pro_com)
            x = FC1(x)
            x = Dense(1,activation='sigmoid')(x)
            dense1.append(x)
        class_out = Average()(dense1)

        # regression task
        dense2 = []
        FC2 = Dense(64, activation='relu')
        for p in np.linspace(0.1,0.5, 5):
            x = Dropout(p)(fc_pro_com)
            x = FC2(x)
            x = Dense(1)(x)
            dense2.append(x)
        regree_out = Average()(dense2)

    class_model = Model(inputs=[protein_input, comp_input], outputs=class_out)
    reg_model = Model(inputs=[protein_input, comp_input], outputs=regree_out)
    plot_model(class_model, to_file=save_dir + "/multitask_training_respectively_class.png")
    plot_model(reg_model, to_file=save_dir + "/multitask_training_respectively_reg.png")
    return  class_model, reg_model
