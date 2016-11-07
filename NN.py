import tensorflow as tf
import numpy as np
import math
# col Num = 14
# 1st col(index is 0)  is ID, ignore it
# 14th col(index is -1, or 13) is Label, so do not use it as a input
NUM_CLASSES = 5
RECORD_SIZE = 12


def inference(records, hidden1_units, hidden2_units):
    '''
    :param records:  Crime record placeholder, from inputs()
    :param hidden1_units: Size of the first hidden layer
    :param hidden2_units: Size of the second hidden layer
    :return: Softmax_linear : Output tensor with the computed logit
    '''
    #Hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([RECORD_SIZE, hidden1_units],
                                stddev=1.0/math.sqrt(float(RECORD_SIZE))),
            name='weights1'
        )

        biases = tf.Variable(
            tf.zeros([hidden1_units]),
            name='biases1'
        )
        hidden1 = tf.nn.relu(tf.matmul(records, weights)+biases)

    #Hidden2
    with tf.name_scope('hidden2'):
        weights=tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0/math.sqrt(float(hidden1_units))),
            name='weights2'
        )
        biases = tf.Variable(
            tf.zeros([hidden2_units]),
            name='biases2'
        )
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights)+biases)


    #Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0/math.sqrt(float(hidden2_units))),
            name='weightsL'
        )
        biases = tf.Variable(
            tf.zeros([NUM_CLASSES]),
            name='biasesL'
        )
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    '''
    :param logits: Logits tensor, float - [batchSize, NUM_CLASSES]
    :param labels: Labels tensor, int32 - [batchSize]
    :return: loss: loss tensor of type float
    '''
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentrophy'
    )
    loss = tf.reduce_mean(cross_entropy, name='xentrophy_mean')
    return loss


def training(loss, learning_rate):
    '''

    :param loss: Loss tensor, from loss()
    :param learning_rate: learnong rate for gradient descent
    :return: train_op: the Op for training
    '''
    # Add a scalar summary for the snapshot loss
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with given learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    '''

    :param logits: Logits tensor, float - [batchSize, NUM_CLASSES]
    :param labels: Labels tensor, int32 - [batchSize], with values in the range [0, NUM_CLASSES).
    :return: a scalar int32 tensor with the number of examples that were predicted correctly, within the batch
    '''

    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))