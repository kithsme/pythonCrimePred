from random import shuffle

import tensorflow as tf
import numpy as np

def read_all_data_for_training():
    xy = np.loadtxt('fullInput.txt', dtype=float)

    shuffled_list = []
    size = xy.__len__()
    randIndexs = range(0, size)
    shuffle(randIndexs)

    for i in randIndexs:
        shuffled_list.append(xy[i])

    return shuffled_list

def read_data_for_training(filename):
    xy = np.loadtxt(filename, dtype=float)

    shuffled_list = []
    size = xy.__len__()
    randIndexs = range(0, size)
    shuffle(randIndexs)

    for i in randIndexs:
        shuffled_list.append(xy[i])

    return shuffled_list

def split_data_set(data_set, train_rate, validate_rate, test_rate):

    num_total = data_set.__len__()

    sum = train_rate + validate_rate + test_rate
    train_rate /= sum
    validate_rate /= sum
    test_rate /= sum
    trainNum = int(train_rate * num_total)
    validateNum = trainNum + int(validate_rate * num_total)

    train_set = data_set[:trainNum]
    validate_set = data_set[trainNum:validateNum]
    test_set = data_set[validateNum:]

    return train_set, validate_set, test_set

def get_feature_label(data_set):

    size = data_set.__len__()
    features = []
    labels = []

    for i in data_set:
        features.append(i[:-1])
        labels.append(i[-1:])

    return features, labels


def read_input_data_for_prediction(filename):
    x = np.loadtxt(filename, dtype=float)
    return x

def generate_test_set(setIdentifier):

    list = read_all_data_for_training()
    print('how many test sets do you want to generate?')
    num = input()

    featureFileName = 'test'
    labelFileName = 'target'
    surfix = '.txt'

    for i in range(1, num+1):
        f_feature = open(setIdentifier+featureFileName+str(i)+surfix,'w')
        f_label = open(setIdentifier+labelFileName+str(i)+surfix,'w')

        size = list.__len__()
        randIndexs = range(0, size)
        shuffle(randIndexs)
        count = 0
        for j in randIndexs:
            if count == 100:
                break
            count+=1
            feature = list[j][:-1]
            label = list[j][-1:]

            feature_string =''
            for fe in range(0,12):
                feature_string += str(int(feature[fe]))
                if fe<11:
                    feature_string+='\t'
            feature_string+='\n'

            label_string=''
            for la in range(0,1):
                label_string += str(int(label[la]))
                label_string += '\n'

            f_feature.write(feature_string)
            f_label.write(label_string)

        f_feature.close()
        f_label.close()

