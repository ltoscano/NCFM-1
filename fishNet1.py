# Author: Park, Han Gyu
# Github: dfyer@github.com
# Contact: gominidive@hotmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Build fish detector based on YOLO-esque neural network

The YOLO architecture is described in https://arxiv.org/abs/1506.02640

Summary of available functions:
     inference: Compute inference on the model inputs to make a prediction
     loss: Compute the loss of the prediction with respect to the labels
"""

from  __future__ import absolute_import
from  __future__ import print_function
from  __future__ import division

import json
from PIL import Image
import numpy as np
import tensorflow as tf

#####
# Some set-ups
Sw = 10
Sh = 6
B = 2
C = 1 #Fish
num_predictions = Sw*Sh*(B*5+1)


def getBoxes(filepath):
    with open(filepath) as json_file:
        return json.load(json_file)


def getIOU(l1, r1, t1, b1, l2, r2, t2, b2):
    l = max(l1, l2)
    r = min(r1, r2)
    t = max(t1, t2)
    b = min(b1, b2)
    return (r-l)*(b-t)


def getImages(boxes, images, labels, l_labels, species, pathfuckedup=False):
    for box in boxes:
        # Get image as np array
        if pathfuckedup:
            origpath = "1280720/" + species + "/" + box['filename'].split("/")[-1]
        else:
            origpath = "1280720/" + species + "/" + box['filename']
        origname = origpath.split('/')[-1]
        orig = Image.open(origpath)

        # Add images
        # Numpy takes the image as row-major indexing (y, x, c)
        # Reference: http://stackoverflow.com/questions/25537137/32-bit-rgba-numpy-array-from-pil-image
        orig_array = np.array(orig)
        images.append(orig_array)

        # Create labels for each cells (128 x 120 pixels each, 10h * 6w cells)
        # For a bounding box, we make a label with size Sh*Sw*(1*5+1)
        label = np.zeros((Sh, Sw, 1*5+1))
        l_idx = np.zeros((Sh, Sw, 1))

        for ann in box['annotations']:
            l_box = ann['x']
            t_box = ann['y']
            w_box = ann['width']
            h_box = ann['height']
            x_box = l_box + w_box/2
            y_box = t_box + h_box/2

            # Find responsible cell
            cell_xidx = int(x_box / 128)
            cell_yidx = int(y_box / 120)

            # Calculate IOU Value
            cell_iou = getIOU(cell_xidx*128, (cell_xidx+1)*128, cell_yidx*120, (cell_yidx+1)*120,
                    x_box, x_box+w_box, y_box, y_box+h_box)

            # Save y, x, h, w, confidence, pr(class_fish)
            # label = [y from the grid cell / 120,
            #          x from the grid cell / 128,
            #          height / 720,
            #          width / 1280,
            #          pr(obj) * iou,
            #          pr(class_fish)
            label[cell_yidx][cell_xidx][0] = (y_box - cell_yidx*120) / 120
            label[cell_yidx][cell_xidx][1] = (x_box - cell_xidx*128) / 128
            label[cell_yidx][cell_xidx][2] = h_box / 720
            label[cell_yidx][cell_xidx][3] = w_box / 1280
            label[cell_yidx][cell_xidx][4] = 1.0 * cell_iou
            label[cell_yidx][cell_xidx][5] = 1.0
            l_idx[cell_yidx][cell_xidx][0] = 1.0
        labels.append(label)
        l_idxs.append(l_idx)



# Get Boxes
alb_boxes = getBoxes("1280720boxes/alb_labels.json")
bet_boxes = getBoxes("1280720boxes/bet_labels.json")
dol_boxes = getBoxes("1280720boxes/dol_labels.json")
lag_boxes = getBoxes("1280720boxes/lag_labels.json")
shark_boxes = getBoxes("1280720boxes/shark_labels.json")
yft_boxes = getBoxes("1280720boxes/yft_labels.json")


# Get images & labels
# Images: N, y, x, c
# Labels: N, y, x, h, w, confidence, pr(C_fish)
images = []
labels = []
l_idxs = []
#getImages(alb_boxes,   images, labels, l_idxs, "ALB")
#getImages(bet_boxes,   images, labels, l_idxs, "BET")
#getImages(dol_boxes,   images, labels, l_idxs, "DOL")
#getImages(lag_boxes,   images, labels, l_idxs, "LAG")
#getImages(shark_boxes, images, labels, l_idxs, "SHARK", True)
getImages(yft_boxes,   images, labels, l_idxs, "YFT", True)

print('images', np.shape(images))
print('labels', np.shape(labels))

#####
# Train / Validation Set
train_ratio = 0.8
pivot_idx = int(len(images) * train_ratio)
train_images = images[:pivot_idx]
train_labels = labels[:pivot_idx]
valid_images = images[pivot_idx:]
valid_labels = labels[pivot_idx:]

#####
# Model hyper-parameters
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
#MOVING_AVERAGE_DECAY = 0.9997
BATCHNORM_EPSILON = 0.001
LEAKY_RELU_ALPHA = 0.1
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

#####
# Operators
def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, strides):
    return tf.nn.conv2d(x, W,
        strides,
        padding='SAME',
        use_cudnn_on_gpu=True)

def batch_norm(x, num_x_depth, phase_trian,
        decay=BATCHNORM_MOVING_AVERAGE_DECAY,
        epsilon=BATCHNORM_EPSILON):
    """ Batch normalization on convolution maps
    Reference: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    """
    with tf.variable_scope('BN'):
        beta = tf.Variable(tf.constant(0.0, shape=[num_x_depth]),
                name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_x_depth]),
                name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

        return normed

def leaky_relu(x):
    return tf.maximum(LEAKY_RELU_ALPHA*x, x)

def max_pool(x, size=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
        strides=[1, size, size, 1], padding='SAME')

#####
# Create the model
def inference(x, y_, y_idx, keep_prob, phase_train):
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    # Convolution Layer 1 with 2x2 Max Pooling
    # Input : 720 x 1280 x  3
    # Filter:   9 x   11      (stride=4)
    # Conv  : 180 x  320 x 64
    # Pool  :  90 x  160 x 64
    with tf.variable_scope('Conv_1'):
        # Don't need bias w/ batch norm (I think)
        end_points['conv1'] = conv2d(x, weight_variables([9, 11, 3, 64]), strides=[1, 4, 4, 1])
        end_points['conv1_bn'] = batch_norm(end_points['conv1'], 64, phase_train)
        end_points['conv1_relu'] = leaky_relu(end_points['conv1_bn'])
        end_points['conv1_pool'] = max_pool(end_points['conv1_relu'])

    # Convolution Layer 2 with 2x2 Max Pooling (#3x3reduce=64, #3x3=128)
    # Input :  90 x  160 x 64
    # Filter:   3 x    3      (stride=1)
    # Conv  :  90 x  160 x 192
    # Pool  :  45 x   80 x 192
    with tf.variable_scope('Conv_2'):
        end_points['conv2_reduce'] = conv2d(end_points['conv1_pool'], weight_variables([1, 1, 64, 64]), strides=[1, 1, 1, 1])
        end_points['conv2'] = conv2d(end_points['conv2_reduce'], weight_variables([3, 3, 64, 192]), strides=[1, 1, 1, 1])
        end_points['conv2_bn'] = batch_norm(end_points['conv2'], 192, phase_train)
        end_points['conv2_relu'] = leaky_relu(end_points['conv2_bn'])
        end_points['conv2_pool'] = max_pool(end_points['conv2_relu'])

    # Convolution Layer 3 with 2x2 Max Pooling
    # Input :  45 x   80 x 192
    # Output:  23 x   40 x 512
    with tf.variable_scope('Conv_3'):
        end_points['conv3_reduce1'] = conv2d(end_points['conv2_pool'], weight_variables([1, 1, 192, 128]), strides=[1, 1, 1, 1])
        end_points['conv3_full1'] = conv2d(end_points['conv3_reduce1'], weight_variables([3, 3, 128, 256]), strides=[1, 1, 1, 1])
        end_points['conv3_bn1'] = batch_norm(end_points['conv3_full1'], 256, phase_train)
        end_points['conv3_relu1'] = leaky_relu(end_points['conv3_bn1'])

        end_points['conv3_reduce2'] = conv2d(end_points['conv3_relu1'], weight_variables([1, 1, 256, 256]), strides=[1, 1, 1, 1])
        end_points['conv3_full2'] = conv2d(end_points['conv3_reduce2'], weight_variables([3, 3, 256, 512]), strides=[1, 1, 1, 1])
        end_points['conv3_bn2'] = batch_norm(end_points['conv3_full2'], 512, phase_train)
        end_points['conv3_relu2'] = leaky_relu(end_points['conv3_bn2'])
        end_points['conv3_pool'] = max_pool(end_points['conv3_relu2'])

    # Convolution Layer 4 with 2x2 Max Pooling
    # Input :  23 x   40 x 512
    # Output:  12 x   20 x 1024
    with tf.variable_scope('Conv_4'):
        end_points['conv4_reduce1'] = conv2d(end_points['conv3_pool'], weight_variables([1, 1, 512, 256]), strides=[1, 1, 1, 1])
        end_points['conv4_full1'] = conv2d(end_points['conv4_reduce1'], weight_variables([3, 3, 256, 512]), strides=[1, 1, 1, 1])
        end_points['conv4_bn1'] = batch_norm(end_points['conv4_full1'], 512, phase_train)
        end_points['conv4_relu1'] = leaky_relu(end_points['conv4_bn1'])

        end_points['conv4_reduce2'] = conv2d(end_points['conv4_relu1'], weight_variables([1, 1, 512, 256]), strides=[1, 1, 1, 1])
        end_points['conv4_full2'] = conv2d(end_points['conv4_reduce2'], weight_variables([3, 3, 256, 512]), strides=[1, 1, 1, 1])
        end_points['conv4_bn2'] = batch_norm(end_points['conv4_full2'], 512, phase_train)
        end_points['conv4_relu2'] = leaky_relu(end_points['conv4_bn2'])

        end_points['conv4_reduce3'] = conv2d(end_points['conv4_relu2'], weight_variables([1, 1, 512, 256]), strides=[1, 1, 1, 1])
        end_points['conv4_full3'] = conv2d(end_points['conv4_reduce3'], weight_variables([3, 3, 256, 512]), strides=[1, 1, 1, 1])
        end_points['conv4_bn3'] = batch_norm(end_points['conv4_full3'], 512, phase_train)
        end_points['conv4_relu3'] = leaky_relu(end_points['conv4_bn3'])

        end_points['conv4_reduce4'] = conv2d(end_points['conv4_relu3'], weight_variables([1, 1, 512, 256]), strides=[1, 1, 1, 1])
        end_points['conv4_full4'] = conv2d(end_points['conv4_reduce4'], weight_variables([3, 3, 256, 512]), strides=[1, 1, 1, 1])
        end_points['conv4_bn4'] = batch_norm(end_points['conv4_full4'], 512, phase_train)
        end_points['conv4_relu4'] = leaky_relu(end_points['conv4_bn4'])

        end_points['conv4_reduce5'] = conv2d(end_points['conv4_relu4'], weight_variables([1, 1, 512, 512]), strides=[1, 1, 1, 1])
        end_points['conv4_full5'] = conv2d(end_points['conv4_reduce5'], weight_variables([3, 3, 512, 1024]), strides=[1, 1, 1, 1])
        end_points['conv4_bn5'] = batch_norm(end_points['conv4_full5'], 1024, phase_train)
        end_points['conv4_relu5'] = leaky_relu(end_points['conv4_bn5'])
        end_points['conv4_pool'] = max_pool(end_points['conv4_relu5'])

    # Convolution Layer 5
    # Input :  23 x   40 x 512
    # Input :   6 x   10 x 1024
    with tf.variable_scope('Conv_5'):
        end_points['conv5_reduce1'] = conv2d(end_points['conv4_pool'], weight_variables([1, 1, 1024, 512]), strides=[1, 1, 1, 1])
        end_points['conv5_full1'] = conv2d(end_points['conv5_reduce1'], weight_variables([3, 3, 512, 1024]), strides=[1, 1, 1, 1])
        end_points['conv5_bn1'] = batch_norm(end_points['conv5_full1'], 1024, phase_train)
        end_points['conv5_relu1'] = leaky_relu(end_points['conv5_bn1'])

        end_points['conv5_reduce2'] = conv2d(end_points['conv5_relu1'], weight_variables([1, 1, 1024, 512]), strides=[1, 1, 1, 1])
        end_points['conv5_full2'] = conv2d(end_points['conv5_reduce2'], weight_variables([3, 3, 512, 1024]), strides=[1, 1, 1, 1])
        end_points['conv5_bn2'] = batch_norm(end_points['conv5_full2'], 1024, phase_train)
        end_points['conv5_relu2'] = leaky_relu(end_points['conv5_bn2'])

        end_points['conv5_full3'] = conv2d(end_points['conv5_relu2'], weight_variables([3, 3, 1024, 1024]), strides=[1, 1, 1, 1])
        end_points['conv5_bn3'] = batch_norm(end_points['conv5_full3'], 1024, phase_train)
        end_points['conv5_relu3'] = leaky_relu(end_points['conv5_bn3'])

        end_points['conv5_full4'] = conv2d(end_points['conv5_relu3'], weight_variables([3, 3, 1024, 1024]), strides=[1, 2, 2, 1])
        end_points['conv5_bn4'] = batch_norm(end_points['conv5_full4'], 1024, phase_train)
        end_points['conv5_relu4'] = leaky_relu(end_points['conv5_bn4'])

    # Convolution Layer 6
    # Input :   6 x   10 x 1024
    # Output:   6 x   10 x 1024
    with tf.variable_scope('Conv_6'):
        end_points['conv6_full1'] = conv2d(end_points['conv5_relu4'], weight_variables([3, 3, 1024, 1024]), strides=[1, 1, 1, 1])
        end_points['conv6_bn1'] = batch_norm(end_points['conv6_full1'], 1024, phase_train)
        end_points['conv6_relu1'] = leaky_relu(end_points['conv6_bn1'])

        end_points['conv6_full2'] = conv2d(end_points['conv6_relu1'], weight_variables([3, 3, 1024, 1024]), strides=[1, 1, 1, 1])
        end_points['conv6_bn2'] = batch_norm(end_points['conv6_full2'], 1024, phase_train)
        end_points['conv6_relu2'] = leaky_relu(end_points['conv6_bn2'])

    # Fully Connected Layer 1
    with tf.variable_scope('FC_1'):
        end_points['fc1_flat'] = tf.reshape(end_points['conv6_relu2'], [-1, 6*10*1024])
        end_points['fc1'] = tf.matmul(end_points['fc1_flat'], weight_variables([6*10*1024, 4096]))
        end_points['fc1_bn'] = batch_norm(end_points['fc1'], 4096, phase_train)
        end_points['fc1_relu'] = tf.nn.relu(end_points['fc1_bn'])
        # Dropout
        end_points['fc1_drop'] = tf.nn.dropout(end_points['fc1_relu'], keep_prob)


    # Fully Connected Layer 2
    with tf.variable_scope('FC_2'):
        end_points['fc2'] = tf.matmul(end_points['fc1_drop'], weight_variables([4096, B*5+1]))
        end_points['fc2_bn'] = batch_norm(end_points['fc2'], b*5+1, phase_train)
        end_points['fc2_relu'] = tf.nn.relu(end_points['fc2_bn'])

    y_pred = tf.reshape(end_points['fc2_relu'], [-1, 6, 10, 1024])
    diff = tf.square(y_pred - y_)
    sqrt_diff = tf.sqare(tf.sqrt(y_pred) - tf.sqrt(y_))
    # Do below on 3-th dimension (0based)
    # for cell in Sh * Sw
    #   for box in B
    #     location_loss += LAMBDA_COORD * (y_idx * ((y_pred[0] - y_[0])**2 + (y_pred[1] - y_[1])**2))
    #     bounding_loss += LAMBDA_COORD * (y_idx * ((y_pred[2] - y_[2])**2 + (y_pred[3] - y_[3])**2))
    #     object_loss += (y_idx * (y_pred[4] - y_[4])**2)
    #     noobj_loss += LAMBDA_NOOBJ * (1 - y_idx) * (y_pred[4] - y_[4])^2
    #   class_loss += (y_idx) * (y_pred[5] - y_[5])^2
    # loss = SIGMA [ all_losses ]

    for name, layer in sorted(end_points.iteritems()):
        if 'pool' in name.split('_'):
            print(name, layer.get_shape())
    print('conv5_relu4', end_points['conv5_relu4'].get_shape())

with tf.Session() as sess:
    # Input variables
    x = tf.placeholder(tf.float32, shape=[None, 720, 1280, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, Sh, Sw, 1*5+1]) # y_ does not look like y_pred!
    y_idx = tf.placeholder(tf.float32, shape=[None, Sh, Sw, 1])
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, accuracy, y_pred = inference(x, y_, y_idx, keep_prob, phase_train)
