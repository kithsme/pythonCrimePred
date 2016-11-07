import os.path
import time
import math

import numpy as np
from numpy import average
from six.moves import xrange
import tensorflow as tf
import NN
import dataManage

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.08, 'Initial learning rate')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 29, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 19, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batchSize', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

def placeholder_inputs(batchSize):
    '''

    :param batchSize:
    :return:
     records_placeholder: crime records placeholder
     labels_placeholer: crime labels placeholder
    '''

    record_placeholder = tf.placeholder(tf.float32, shape=(batchSize, NN.RECORD_SIZE))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batchSize))
    return record_placeholder, labels_placeholder

def fill_feed_dict(data_set, cur_batch, batch_size, records_pl, labels_pl):
    '''

    :param records_pl: the records placeholder from placeholder.input()
    :param labels_pl: the labels placeholder from placeholder.input()
    :return: feed_dict: the feed dictionary mapping from placeholders to values
    '''

    # Create the feed_dict for the placeholders filled with the next batch size examples
    size = data_set.__len__()

    if cur_batch + batch_size <= size:
        features_feed, labels_feed_list = dataManage.get_feature_label(data_set[cur_batch:cur_batch+batch_size])
        cur_batch += batch_size
    else:
        features_feed, labels_feed_list = dataManage.get_feature_label(data_set[cur_batch:])
        remain = features_feed.__len__()
        features_feed2, labels_feed_list2 = dataManage.get_feature_label(data_set[:(batch_size-remain)])
        features_feed = features_feed + features_feed2
        labels_feed_list = labels_feed_list + labels_feed_list2
        cur_batch = batch_size-remain

    labels_feed = []
    for i in labels_feed_list:
        labels_feed.append(i[0])

    feed_dict = {
        records_pl: features_feed,
        labels_pl: labels_feed,
    }
    return feed_dict, cur_batch


def do_eval(sess,
            eval_correct,
            records_pl,
            labels_pl,
            data_set):
    '''

    :param sess: the session
    :param eval_correct: the tensor that returns the number of the correct predictions
    :param records_pl: crime records placeholder
    :param labels_pl: crime labels placeholder
    :param data_set: the set of records and labels to evaluate from input_data.read_data_sets()
    :return:
    '''

    batch_size = FLAGS.batchSize
    size = data_set.__len__()

    while size < batch_size:
        data_set = data_set+data_set
        size = data_set.__len__()

    # And run one epoch of eval
    true_count = 0 # Counts the number of correct predictions
    steps_per_epoch = data_set.__len__() // batch_size
    num_examples = steps_per_epoch * batch_size
    cur_batch = 0
    for step in xrange(steps_per_epoch):
        feed_dict, cur_batch = fill_feed_dict(data_set, cur_batch, batch_size,
                                   records_pl,
                                   labels_pl)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count)/float(num_examples)
    print(' Num examples: %d Num correct: %d Precision: %0.04f' %
          (num_examples, true_count, precision))




def run_training(trainingFile):

    data_set = dataManage.read_data_for_training(trainingFile)

    train_data_set, validate_data_set, test_data_set = dataManage.split_data_set(data_set=data_set,
                                                                                 train_rate=0.8,
                                                                                 validate_rate=0.05,
                                                                                 test_rate=0.15)
    caseZeroSet = []
    caseOneSet = []
    caseTwoSet = []
    caseThreeSet = []
    caseFourSet = []

    for i in data_set:
        if i[-1]==0:
            caseZeroSet.append(i)
        elif i[-1]==1:
            caseOneSet.append(i)
        elif i[-1]==2:
            caseTwoSet.append(i)
        elif i[-1]==3:
            caseThreeSet.append(i)
        else:
            caseFourSet.append(i)

    with tf.Graph().as_default():
        records_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batchSize)

        logits = NN.inference(records_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        loss = NN.loss(logits, labels_placeholder)

        train_op = NN.training(loss, FLAGS.learning_rate)

        eval_correct = NN.evaluation(logits, labels_placeholder)

        summary = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        sess.run(init)
        cur_batch=0

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict, cur_batch = fill_feed_dict(train_data_set, cur_batch, FLAGS.batchSize,
                                       records_placeholder,
                                       labels_placeholder)

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)


            duration = time.time()-start_time

            if step % 500 == 0:
                print('Step %d: loss= %.2f (%.3f sec)' % (step, loss_value, duration))

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step+1) % 1000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

        else:
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    train_data_set)

            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    validate_data_set)

            print('Test Data Eval:')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    test_data_set)
            '''
            print('Test case=0 (Theft):')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    caseZeroSet)

            print('Test case=1 (Assault):')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    caseOneSet)

            print('Test case=2 (Robbery):')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    caseTwoSet)

            print('Test case=3 (Sexual Harassment):')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    caseThreeSet)

            print('Test case=4 (Murder):')
            do_eval(sess,
                    eval_correct,
                    records_placeholder,
                    labels_placeholder,
                    caseFourSet)'''

        '''exportname = trainingFile.replace('.txt', '')
        w1, b1, w2, b2, wl, b1 = sess.run(tf.weights_.eval())
        np.savetxt("W1.csv", w1, delimiter=",")
        np.savetxt("B1.csv", b1, delimiter=",")
        np.savetxt("W2.csv", w2, delimiter=",")
        np.savetxt("B2.csv", b2, delimiter=",")
        np.savetxt("WL.csv", wl, delimiter=",")
        np.savetxt("BL.csv", bl, delimiter=",")'''

        trainable_vars= tf.trainable_variables()
        varSaver = tf.train.Saver({'weights1': trainable_vars[0], 'biases1': trainable_vars[1],
                               'weights2': trainable_vars[2], 'biases2': trainable_vars[3],
                                'weightsLinear':trainable_vars[4], 'biasesLinear': trainable_vars[5]})

        exportname= trainingFile.replace('.txt', '')
        save_path = varSaver.save(sess, "/tmp/model"+exportname+".ckpt")

    with sess.as_default():

        weights1 = trainable_vars[0].eval()
        biases1 = trainable_vars[1].eval()
        weights2 = trainable_vars[2].eval()
        biases2 = trainable_vars[3].eval()
        weightsL = trainable_vars[4].eval()
        biasesL = trainable_vars[5].eval()

        np.savetxt(exportname+'_weights1.csv', weights1, delimiter=",")
        np.savetxt(exportname+'_biases1.csv', biases1, delimiter=",")
        np.savetxt(exportname+'_weights2.csv', weights2, delimiter=",")
        np.savetxt(exportname+'_biases2.csv', biases2, delimiter=",")
        np.savetxt(exportname+'_weightsL.csv', weightsL, delimiter=",")
        np.savetxt(exportname+'_biasesL.csv', biasesL, delimiter=",")


        a=3

    return sess, logits, records_placeholder


def decision_tree(x):
    pat = -1
    if x[7] == 2:
        return 1
    else:
        if x[1] in [1,3,7,8,9,10,11,12]:
            if x[9] == 2:
                if x[3] == 2:
                    return 6
                else:
                    return 5
            else:
                return 3
        elif x[1] == 2:
            return 2
        else:
            if x[9] == 2:
                if x[2] in [2,3]:
                    return 8
                else:
                    return 7
            else:
                return 4

    if pat == -1:
        raise 'eqwe!'

def pattern_retrieve(xs):

    pats = []
    for x in xs:
        pat = decision_tree(x)
        pats.append(pat)

    return pats

def do_pred(xs, i, sess, logits, records_pl):

    #print(sess.run(logits[i], feed_dict={records_pl: xs}))
    aa = sess.run(logits[i], feed_dict={records_pl: xs})

    maxVal = -10000.0
    maxInd = -1

    for a in range(0, 5):
        if aa[a] > maxVal:
            maxVal = aa[a]
            maxInd = a

    return maxInd

def type_from_pattern(pats, xs, sess1, logits1, records_pl1, sess5, logits5, records_pl5):

    ys = []

    for i in range(0, pats.__len__()):
        if pats[i] == 1:
            ys.append(do_pred(xs, i, sess1, logits1, records_pl1))
        elif pats[i] == 5:
            ys.append(do_pred(xs, i, sess5, logits5, records_pl5))
        elif pats[i] in [3,4,5,8]:
            ys.append(1)
        else:
            ys.append(3)

    return ys

def evaluate_set(num, set_identifier):

    testSetScore = []

    for i in range(1, num+1):
        f_target = open(set_identifier+'target'+str(i)+'.txt','r')
        f_result = open(set_identifier+'result'+str(i)+'.txt','r')

        targets = f_target.readlines()
        results = f_result.readlines()

        size = targets.__len__()
        correct = 0
        for j in range(0, size):
            if targets[j] == results[j]:
                correct +=1

        score = float(correct)/float(size)
        testSetScore.append(score)

    print(testSetScore)
    print('min: %.3f avg: %.3f' % (min(testSetScore), average(testSetScore)))

    f = open(set_identifier+'summary.txt', 'w')
    f.write('min: %.3f avg: %.3f' % (min(testSetScore), average(testSetScore)))
    f.write('\n')
    for i in range(0, testSetScore.__len__()):
        f.write(str(testSetScore[i])+'\t')
    f.write('\n')
    f.close()

def evaluate_each_case(set_identifier, sess1, logits1, records_pl1, sess5, logits5, records_pl5):

    xy_0 = dataManage.read_data_for_training('case0.txt')
    xy_0 = xy_0[0:FLAGS.batchSize]
    x_0, y_0 = dataManage.get_feature_label(xy_0)
    pats_in_case0 = pattern_retrieve(x_0)
    ys_in_case0 = type_from_pattern(pats_in_case0, x_0, sess1, logits1, records_pl1, sess5, logits5, records_pl5)
    size_0 = xy_0.__len__()
    hit_0 = 0
    for i in range(0, size_0):
        if ys_in_case0[i]==0:
            hit_0+=1

    xy_1 = dataManage.read_data_for_training('case1.txt')
    xy_1 = xy_1[0:FLAGS.batchSize]
    x_1, y_1 = dataManage.get_feature_label(xy_1)
    pats_in_case1 = pattern_retrieve(x_1)
    ys_in_case1 = type_from_pattern(pats_in_case1, x_1, sess1, logits1, records_pl1, sess5, logits5, records_pl5)
    size_1 = xy_1.__len__()
    hit_1 = 0
    for i in range(0, size_1):
        if ys_in_case1[i] == 1:
            hit_1 += 1

    xy_2 = dataManage.read_data_for_training('case2.txt')
    xy_2 = xy_2[0:FLAGS.batchSize]
    x_2, y_2 = dataManage.get_feature_label(xy_2)
    pats_in_case2 = pattern_retrieve(x_2)
    ys_in_case2 = type_from_pattern(pats_in_case2, x_2, sess1, logits1, records_pl1, sess5, logits5, records_pl5)
    size_2 = xy_2.__len__()
    hit_2 = 0
    for i in range(0, size_2):
        if ys_in_case2[i] == 2:
            hit_2 += 1

    xy_3 = dataManage.read_data_for_training('case3.txt')
    xy_3 = xy_3[0:FLAGS.batchSize]
    x_3, y_3 = dataManage.get_feature_label(xy_3)
    pats_in_case3 = pattern_retrieve(x_3)
    ys_in_case3 = type_from_pattern(pats_in_case3, x_3, sess1, logits1, records_pl1, sess5, logits5, records_pl5)
    size_3 = xy_3.__len__()
    hit_3 = 0
    for i in range(0, size_3):
        if ys_in_case3[i] == 3:
            hit_3 += 1

    xy_4 = dataManage.read_data_for_training('case4.txt')
    xy_4 = xy_4[0:FLAGS.batchSize]
    x_4, y_4 = dataManage.get_feature_label(xy_4)
    pats_in_case4 = pattern_retrieve(x_4)
    ys_in_case4 = type_from_pattern(pats_in_case4, x_4, sess1, logits1, records_pl1, sess5, logits5, records_pl5)
    size_4 = xy_4.__len__()
    hit_4 = 0
    for i in range(0, size_4):
        if ys_in_case4[i] == 4:
            hit_4 += 1

    print('case0 (theft): %.2f' % (float(hit_0)/float(size_0)))
    print('case1 (assault): %.2f' % (float(hit_1)/float(size_1)))
    print('case2 (robbery): %.2f' % (float(hit_2)/float(size_2)))
    print('case3 (sex. harass): %.2f' % (float(hit_3)/float(size_3)))
    print('case4 (murder): %.2f' % (float(hit_4)/float(size_4)))

    f = open(set_identifier+'caseSummary.txt', 'w')
    f.write('case0 (theft): %.2f \n' % (float(hit_0)/float(size_0)))
    f.write('case1 (assault): %.2f \n' % (float(hit_1)/float(size_1)))
    f.write('case2 (robbery): %.2f \n' % (float(hit_2)/float(size_2)))
    f.write('case3 (sex. harass): %.2f \n' % (float(hit_3)/float(size_3)))
    f.write('case4 (murder): %.2f \n' % (float(hit_4)/float(size_4)))
    f.close()


def main(_):
    print('specify your set identifier')
    set_identifier = raw_input()
    if not (set_identifier=='') and not (set_identifier==None):
        set_identifier += '_'

    dataManage.generate_test_set(set_identifier)
    print('how many test sets do you want to run? (Please enter randomly selected 100 records)')
    numFiles = input()
    inputFilenamePrefix = 'test'
    inputFilenameSurfix = '.txt'

    outputFilenamePrefix = 'result'
    outputFilenameSurfix = '.txt'

    sess1, logits1, records_pl1 = run_training("package1.txt")
    sess5, logits5, records_pl5 = run_training("package2.txt")

    xs_in_all_file = []
    ys_in_all_file = []
    for a in range(1, numFiles+1):
        xs_in_one_file = dataManage.read_input_data_for_prediction(set_identifier+inputFilenamePrefix+str(a)+inputFilenameSurfix)
        pats_in_one_file = pattern_retrieve(xs_in_one_file)
        ys_in_one_file = type_from_pattern(pats_in_one_file, xs_in_one_file,
                                           sess1, logits1, records_pl1, sess5, logits5, records_pl5)
        f = open(set_identifier+outputFilenamePrefix+str(a)+outputFilenameSurfix, 'w')
        print('Generate results %d / %d' % (a, numFiles))
        for b in range(0, ys_in_one_file.__len__()):
            f.write(str(ys_in_one_file[b])+'\n')
        f.close()

    evaluate_set(numFiles, set_identifier)

    evaluate_each_case(set_identifier, sess1, logits1, records_pl1, sess5, logits5, records_pl5)


if __name__ == '__main__':
    tf.app.run()