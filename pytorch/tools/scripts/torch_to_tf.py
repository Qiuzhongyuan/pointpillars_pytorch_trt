# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util
import pickle
from tensorflow.python.platform import gfile
import tensorflow.contrib as contrib


def conv_op(x, weight, conv_mode="conv", stride=1, name='0', bias=None, rate=1):
    if conv_mode == "depthwise":
        weight = np.transpose(weight, (2, 3, 0, 1))  # 调换为TensorFlow的weight格式
    else:
        weight = np.transpose(weight, (2, 3, 1, 0))  # 调换为TensorFlow的weight格式
    with tf.variable_scope(name):
        init = tf.constant_initializer(weight)
        weight = tf.get_variable('W', shape=weight.shape, initializer=init)
        if conv_mode == "deconv":
            # batch_size = x.shape[0]
            out_height, out_width = x.shape[1] * 2, x.shape[2] * 2
            num_output_channels = weight.shape[2]
            output_shape = tf.stack([1, out_height, out_width, num_output_channels], axis=0)
            # x = tf.nn.depthwise_conv2d(x,weight,strides=(1,stride, stride, 1),padding='SAME',rate=None,name=None,data_format=None)
            # x = tf.nn.depthwise_conv2d_native(x, weight, (1,stride, stride, 1), padding='SAME')
            x = tf.nn.conv2d_transpose(x, weight, output_shape=output_shape, strides=(1, 2, 2, 1), padding='SAME')
            # 1 torch.Size([1, 32, 64, 128]) 2 torch.Size([1, 32, 128, 256])
        elif conv_mode == "conv":
            # x = tf.layers.conv2d(x,out_nc,kernel,strides=stride, padding='SAME',kernel_initializer=init,use_bias=False)
            x = tf.nn.conv2d(x, weight, strides=(1, stride, stride, 1), padding='SAME')
        elif conv_mode == "dilation_conv":
            x = tf.nn.atrous_conv2d(x, weight, rate=rate, padding='SAME')
        if bias is not None:
            init = tf.constant_initializer(bias)
            b = tf.get_variable('bias', shape=bias.shape, initializer=init)
            x = tf.nn.bias_add(x, b)
        return x


def bn(x, bn_weight, name='0', variance_epsilon=0.00001):
    # bn_weight: (4, out_nc)四行，代表weight, bias, running_mean, running_var
    with tf.variable_scope(name):
        center_init = tf.constant_initializer(bn_weight[1])
        scale_init = tf.constant_initializer(bn_weight[0])
        moving_mean_init = tf.constant_initializer(bn_weight[2])
        moving_var_init = tf.constant_initializer(bn_weight[3])

        scale = tf.get_variable('scale', shape=(bn_weight.shape[1],), initializer=scale_init)
        offset = tf.get_variable('offset', shape=(bn_weight.shape[1],), initializer=center_init)
        mean = tf.get_variable('mean', shape=(bn_weight.shape[1],), initializer=moving_mean_init)
        variance = tf.get_variable('variance', shape=(bn_weight.shape[1],), initializer=moving_var_init)
        x = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=variance_epsilon, name=name)
    return x


def conv_bn_relu(x, weights, conv_name, bn_name, stride, name):
    with tf.variable_scope(name):
        conv_weight = weights[conv_name]
        x = conv_op(x, conv_weight, conv_mode="conv", stride=stride, name='conv', bias=None)
        if bn_name is not None:
            bn_weight = np.stack([weights[bn_name + '.weight'],
                                  weights[bn_name + '.bias'],
                                  weights[bn_name + '.running_mean'],
                                  weights[bn_name + '.running_var']])

            x = bn(x, bn_weight, name='bn')
            x = tf.nn.relu6(x, name='relu')
    return x


def block0(x, weights, stride=2, name='block0'):
    with tf.variable_scope(name):
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.0.1.weight', 'backbone_2d.blocks.0.2', 1, 'conv_bn_relu')
    return x


def block1(x, weights, stride=2, name='block1'):
    with tf.variable_scope(name):
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.1.0.weight', 'backbone_2d.blocks.1.1', stride, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.1.3.weight', 'backbone_2d.blocks.1.4', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.1.6.weight', 'backbone_2d.blocks.1.7', 1, 'conv_bn_relu_3')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.1.9.weight', 'backbone_2d.blocks.1.10', 1, 'conv_bn_relu_4')
    return x

def block2(x, weights, stride=2, name='block2'):
    with tf.variable_scope(name):
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.2.0.weight', 'backbone_2d.blocks.2.1', stride, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.2.3.weight', 'backbone_2d.blocks.2.4', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.2.6.weight', 'backbone_2d.blocks.2.7', 1, 'conv_bn_relu_3')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.2.9.weight', 'backbone_2d.blocks.2.10', 1, 'conv_bn_relu_4')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.2.12.weight', 'backbone_2d.blocks.2.13', 1, 'conv_bn_relu_5')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.2.15.weight', 'backbone_2d.blocks.2.16', 1, 'conv_bn_relu_6')
    return x

def block3(x, weights, stride=2, name='block3'):
    with tf.variable_scope(name):
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.3.0.weight', 'backbone_2d.blocks.3.1', stride, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.3.3.weight', 'backbone_2d.blocks.3.4', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.3.6.weight', 'backbone_2d.blocks.3.7', 1, 'conv_bn_relu_3')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.3.9.weight', 'backbone_2d.blocks.3.10', 1, 'conv_bn_relu_4')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.3.12.weight', 'backbone_2d.blocks.3.13', 1, 'conv_bn_relu_5')
        x = conv_bn_relu(x, weights, 'backbone_2d.blocks.3.15.weight', 'backbone_2d.blocks.3.16', 1, 'conv_bn_relu_6')
    return x

def deblocks():
    pass


def dilation_conv(x, weights, conv_name, stride, name='0', rate=2, bn_name=None):
    with tf.variable_scope(name):
        conv_weight = weights[conv_name]
        x = conv_op(x, conv_weight, conv_mode="dilation_conv", stride=stride, name='0', bias=None, rate=rate)
        # x = conv_op(x, conv_weight, conv_mode="dilation_conv", stride=stride, name='0', bias=None, rate=rate)
    if bn_name is not None:
        bn_weight = np.stack([weights[bn_name + '.weight'],
                              weights[bn_name + '.bias'],
                              weights[bn_name + '.running_mean'],
                              weights[bn_name + '.running_var']])
        x = bn(x, bn_weight, name='bn_' + name)
        x = tf.nn.relu6(x, name='relu_' + name)
    return x


def block2(x, weights, stride=2, name='block2'):
    with tf.variable_scope(name):
        x1 = x
        x = conv_bn_relu(x, weights, 'regular2_1.ext1_conv.weight', 'regular2_1.ext1_bn', 1, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'regular2_1.ext2_conv.weight', 'regular2_1.ext2_bn', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'regular2_1.ext3_conv.weight', 'regular2_1.ext3_bn', 1, 'conv_bn_relu_3')
        x = x1 + x
        x2 = x
        x = conv_bn_relu(x, weights, 'dilated2_2.ext1_conv.weight', 'dilated2_2.ext1_bn', 1, 'conv_bn_relu_4')
        # dilation conv
        x = dilation_conv(x, weights, "dilated2_2.ext2_conv.weight", 1, name='1', rate=2, bn_name='dilated2_2.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated2_2.ext3_conv.weight', 'dilated2_2.ext3_bn', 1, 'conv_bn_relu_6')
        x = x2 + x
        x3 = x
        x = conv_bn_relu(x, weights, 'asymmetric2_3.ext1_conv.weight', 'asymmetric2_3.ext1_bn', 1, 'conv_bn_relu_7')
        x = conv_bn_relu(x, weights, 'asymmetric2_3.ext2_conv.weight', 'asymmetric2_3.ext2_bn', 1, 'conv_bn_relu_8')
        x = conv_bn_relu(x, weights, 'asymmetric2_3.ext3_conv.weight', 'asymmetric2_3.ext3_bn', 1, 'conv_bn_relu_9')
        x = x3 + x
        x4 = x
        x = conv_bn_relu(x, weights, 'dilated2_4.ext1_conv.weight', 'dilated2_4.ext1_bn', 1, 'conv_bn_relu_10')
        x = dilation_conv(x, weights, "dilated2_4.ext2_conv.weight", 1, name='2', rate=4, bn_name='dilated2_4.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated2_4.ext3_conv.weight', 'dilated2_4.ext3_bn', 1, 'conv_bn_relu_12')
        x = x4 + x
    return x


def block3(x, weights, stride=2, name='block3'):
    with tf.variable_scope(name):
        x1 = x
        x = conv_bn_relu(x, weights, 'regular2_5.ext1_conv.weight', 'regular2_5.ext1_bn', 1, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'regular2_5.ext2_conv.weight', 'regular2_5.ext2_bn', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'regular2_5.ext3_conv.weight', 'regular2_5.ext3_bn', 1, 'conv_bn_relu_3')
        x = x1 + x
        x2 = x
        x = conv_bn_relu(x, weights, 'dilated2_6.ext1_conv.weight', 'dilated2_6.ext1_bn', 1, 'conv_bn_relu_4')
        # dilation conv
        x = dilation_conv(x, weights, "dilated2_6.ext2_conv.weight", 1, name='3', rate=8, bn_name='dilated2_6.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated2_6.ext3_conv.weight', 'dilated2_6.ext3_bn', 1, 'conv_bn_relu_6')
        x = x2 + x
        x3 = x
        x = conv_bn_relu(x, weights, 'asymmetric2_7.ext1_conv.weight', 'asymmetric2_7.ext1_bn', 1, 'conv_bn_relu_7')
        x = conv_bn_relu(x, weights, 'asymmetric2_7.ext2_conv.weight', 'asymmetric2_7.ext2_bn', 1, 'conv_bn_relu_8')
        x = conv_bn_relu(x, weights, 'asymmetric2_7.ext3_conv.weight', 'asymmetric2_7.ext3_bn', 1, 'conv_bn_relu_9')
        x = x3 + x
        x4 = x
        x = conv_bn_relu(x, weights, 'dilated2_8.ext1_conv.weight', 'dilated2_8.ext1_bn', 1, 'conv_bn_relu_10')
        # dilation conv
        x = dilation_conv(x, weights, "dilated2_8.ext2_conv.weight", 1, name='4', rate=16, bn_name='dilated2_8.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated2_8.ext3_conv.weight', 'dilated2_8.ext3_bn', 1, 'conv_bn_relu_12')
        x = x4 + x
    return x


def block4(x, weights, stride=2, name='block4'):
    with tf.variable_scope(name):
        x1 = x
        x = conv_bn_relu(x, weights, 'regular3_0.ext1_conv.weight', 'regular3_0.ext1_bn', 1, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'regular3_0.ext2_conv.weight', 'regular3_0.ext2_bn', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'regular3_0.ext3_conv.weight', 'regular3_0.ext3_bn', 1, 'conv_bn_relu_3')
        x = x1 + x
        x2 = x
        x = conv_bn_relu(x, weights, 'dilated3_1.ext1_conv.weight', 'dilated3_1.ext1_bn', 1, 'conv_bn_relu_4')
        # dilation conv
        x = dilation_conv(x, weights, "dilated3_1.ext2_conv.weight", 1, name='5', rate=2, bn_name='dilated3_1.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated3_1.ext3_conv.weight', 'dilated3_1.ext3_bn', 1, 'conv_bn_relu_6')
        x = x2 + x
        x3 = x
        x = conv_bn_relu(x, weights, 'asymmetric3_2.ext1_conv.weight', 'asymmetric3_2.ext1_bn', 1, 'conv_bn_relu_7')
        x = conv_bn_relu(x, weights, 'asymmetric3_2.ext2_conv.weight', 'asymmetric3_2.ext2_bn', 1, 'conv_bn_relu_8')
        x = conv_bn_relu(x, weights, 'asymmetric3_2.ext3_conv.weight', 'asymmetric3_2.ext3_bn', 1, 'conv_bn_relu_9')
        x = x3 + x
        x4 = x
        x = conv_bn_relu(x, weights, 'dilated3_3.ext1_conv.weight', 'dilated3_3.ext1_bn', 1, 'conv_bn_relu_10')
        # dilation conv
        x = dilation_conv(x, weights, "dilated3_3.ext2_conv.weight", 1, name='6', rate=4, bn_name='dilated3_3.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated3_3.ext3_conv.weight', 'dilated3_3.ext3_bn', 1, 'conv_bn_relu_12')
        x = x4 + x
    return x


def block5(x, weights, stride=2, name='block5'):
    with tf.variable_scope(name):
        x1 = x
        x = conv_bn_relu(x, weights, 'regular3_4.ext1_conv.weight', 'regular3_4.ext1_bn', 1, 'conv_bn_relu_1')
        x = conv_bn_relu(x, weights, 'regular3_4.ext2_conv.weight', 'regular3_4.ext2_bn', 1, 'conv_bn_relu_2')
        x = conv_bn_relu(x, weights, 'regular3_4.ext3_conv.weight', 'regular3_4.ext3_bn', 1, 'conv_bn_relu_3')
        x = x1 + x
        x2 = x
        x = conv_bn_relu(x, weights, 'dilated3_5.ext1_conv.weight', 'dilated3_5.ext1_bn', 1, 'conv_bn_relu_4')
        # dilation conv
        x = dilation_conv(x, weights, "dilated3_5.ext2_conv.weight", 1, name='7', rate=8, bn_name='dilated3_5.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated3_5.ext3_conv.weight', 'dilated3_5.ext3_bn', 1, 'conv_bn_relu_6')
        x = x2 + x
        x3 = x
        x = conv_bn_relu(x, weights, 'asymmetric3_6.ext1_conv.weight', 'asymmetric3_6.ext1_bn', 1, 'conv_bn_relu_7')
        x = conv_bn_relu(x, weights, 'asymmetric3_6.ext2_conv.weight', 'asymmetric3_6.ext2_bn', 1, 'conv_bn_relu_8')
        x = conv_bn_relu(x, weights, 'asymmetric3_6.ext3_conv.weight', 'asymmetric3_6.ext3_bn', 1, 'conv_bn_relu_9')
        x = x3 + x
        x4 = x
        x = conv_bn_relu(x, weights, 'dilated3_7.ext1_conv.weight', 'dilated3_7.ext1_bn', 1, 'conv_bn_relu_10')
        # dilation conv
        x = dilation_conv(x, weights, "dilated3_7.ext2_conv.weight", 1, name='8', rate=16, bn_name='dilated3_7.ext2_bn')
        x = conv_bn_relu(x, weights, 'dilated3_7.ext3_conv.weight', 'dilated3_7.ext3_bn', 1, 'conv_bn_relu_12')
        x = x4 + x
    return x


def block6(x, weights, stride=1, name='block6'):
    with tf.variable_scope(name):
        x1 = x
        x = conv_bn_relu(x, weights, 'upsample4_0.main_conv.weight', 'upsample4_0.main_bn', 1, 'conv_bn_relu_1')
        x = tf.image.resize_images(x, size=(x.shape[1] * 2, x.shape[2] * 2), method=1)

        x2 = x
        x = conv_bn_relu(x1, weights, 'upsample4_0.ext1_conv.weight', 'upsample4_0.ext1_bn', 1, 'conv_bn_relu_2')
        x = deconv(x, weights, 'upsample4_0.ext2_conv.weight', stride=2, name="9", bn_name='upsample4_0.ext2_bn')
        x = conv_bn_relu(x, weights, 'upsample4_0.ext3_conv.weight', 'upsample4_0.ext3_bn', 1, 'conv_bn_relu_4')
        x = x + x2
        x3 = x
        x = conv_bn_relu(x, weights, 'regular4_1.ext1_conv.weight', 'regular4_1.ext1_bn', 1, 'conv_bn_relu_5')
        x = conv_bn_relu(x, weights, 'regular4_1.ext2_conv.weight', 'regular4_1.ext2_bn', 1, 'conv_bn_relu_6')
        x = conv_bn_relu(x, weights, 'regular4_1.ext3_conv.weight', 'regular4_1.ext3_bn', 1, 'conv_bn_relu_7')
        x = x + x3
        x4 = x
        x = conv_bn_relu(x, weights, 'regular4_2.ext1_conv.weight', 'regular4_2.ext1_bn', 1, 'conv_bn_relu_8')
        x = conv_bn_relu(x, weights, 'regular4_2.ext2_conv.weight', 'regular4_2.ext2_bn', 1, 'conv_bn_relu_9')
        x = conv_bn_relu(x, weights, 'regular4_2.ext3_conv.weight', 'regular4_2.ext3_bn', 1, 'conv_bn_relu_10')
        x = x + x4
    return x


def block7(x, weights, stride=1, name='block7'):
    with tf.variable_scope(name):
        x1 = x
        x = conv_bn_relu(x, weights, 'upsample5_0.main_conv.weight', 'upsample5_0.main_bn', 1, 'conv_bn_relu_1')
        x = tf.image.resize_images(x, size=(x.shape[1] * 2, x.shape[2] * 2), method=1)
        x2 = x
        x = conv_bn_relu(x1, weights, 'upsample5_0.ext1_conv.weight', 'upsample5_0.ext1_bn', 1, 'conv_bn_relu_2')
        x = deconv(x, weights, 'upsample5_0.ext2_conv.weight', stride=2, name="10", bn_name='upsample5_0.ext2_bn')
        x = conv_bn_relu(x, weights, 'upsample5_0.ext3_conv.weight', 'upsample5_0.ext3_bn', 1, 'conv_bn_relu_4')
        x = x + x2
        x3 = x
        x = conv_bn_relu(x, weights, 'regular5_1.ext1_conv.weight', 'regular5_1.ext1_bn', 1, 'conv_bn_relu_5')
        x = conv_bn_relu(x, weights, 'regular5_1.ext2_conv.weight', 'regular5_1.ext2_bn', 1, 'conv_bn_relu_6')
        x = conv_bn_relu(x, weights, 'regular5_1.ext3_conv.weight', 'regular5_1.ext3_bn', 1, 'conv_bn_relu_7')
        x = x + x3
    return x


def enet_model(x, weights):
    with tf.variable_scope('enet'):
        # x = tf.reshape(x, (-1, 512, 1024, 3), name='input')
        x = block0(x, weights, stride=2, name='block1')
        x = block1(x, weights, stride=2, name='block1')
        x = block2(x, weights, stride=2, name='block2')
        x = block3(x, weights, stride=2, name='block3')
        x = block4(x, weights, stride=2, name='block4')
        x = block5(x, weights, stride=2, name='block5')
        x = block6(x, weights, stride=1, name='block6')
        x = block7(x, weights, stride=1, name='block7')
        x = deconv(x, weights, "transposed_conv.weight", stride=2, name="output")
    return x


def deconv(x, weights, conv_name, stride, name, bn_name=None):
    with tf.variable_scope(name):
        conv_weight = weights[conv_name]
        x = conv_op(x, conv_weight, conv_mode="deconv", stride=stride, name='0', bias=None)
    if bn_name is not None:
        bn_weight = np.stack([weights[bn_name + '.weight'],
                              weights[bn_name + '.bias'],
                              weights[bn_name + '.running_mean'],
                              weights[bn_name + '.running_var']])
        x = bn(x, bn_weight, name='bn')
        x = tf.nn.relu6(x, name='relu')
    return x


def enet(weights, save_path):
    image = tf.placeholder(
        shape=[None, 416, 768, 3],
        dtype=tf.float32,
        name='input')

    model = enet_model(image, weights)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    images = np.ones((1, 416, 768, 3)).astype(np.float32)
    images = np.load('inputs.npy')
    data = sess.run(model, feed_dict={image: images})
    # print([n.name for n in sess.graph_def.node])

    # print(data)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [
        'enet/output/0/conv2d_transpose'])  # enet/output/0/conv2d_transpose
    with tf.gfile.FastGFile(save_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    torch_out = np.load('outputs.npy')

    error = np.abs(torch_out - data).sum()
    print('error:', error)


if __name__ == "__main__":
    weights_pkl = "./weights.pkl"
    with open(weights_pkl, 'rb') as f:
        weights = pickle.load(f)

    # for k, v in weights.items():
    #     print(k, v.shape)
    pb_path = "./enet.pb"
    enet(weights, pb_path)
