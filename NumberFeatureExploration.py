import os

import numpy as np
from matplotlib import pyplot as plt

from src.CNN_v1 import CNN
from src.Classifier import Classifier
from src.DataHandler import load_data, STD_SCALER, reshuffle
from src.FeaturesSelector import FeaturesSelector

################################################################################
################################## PARAMETERS ##################################
################################################################################

# Number of attempts that have to be averaged
NUM_ATTEMPTS = 5
USE_CNN = True

linear_dict = {
    'fit_intercept': True,
    'normalize': False,
    'copy_X': True,
    'n_jobs': None
}

lor_dict = {
    'penalty': 'l2',
    'dual': False,
    'tol': 1e-4,
    'C': 0.5,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'class_weight': None,
    'random_state': None,
    'solver': 'warn',
    'max_iter': 100,
    'multi_class': 'auto',
    'verbose': 0,
    'warm_start': False,
    'n_jobs': None
}

svm_dict = {
    'C': 1.0,
    'kernel': 'rbf',
    'degree': 2,
    'gamma': 'auto',
    'coef0': 0.0,
    'shrinking': True,
    'probability': False,
    'tol': 0.001,
    'cache_size': 200,
    'class_weight': None,
    'verbose': False,
    'max_iter': -1,
    'decision_function_shape': 'ovr',
    'random_state': None,
}

gnb_dict = {
    'priors': None,
    'var_smoothing': 1e-9
}

lda_dict = {
    'solver': 'lsqr',
    'shrinkage': None,
    'priors': None,
    'n_components': None,
    'store_covariance': False,
    'tol': 0.0001,
}

qda_dict = {
    'priors': None,
    'reg_param': 0.0,
    'store_covariance': False,
    'tol': 0.0001,
}

nn_dict = {
    'batch_size': 128,
    'epochs': 15,
    'verbose': 0,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

################################################################################
#################################### SCRIPT ####################################
################################################################################

# Test listed classifiers. Find optimal features for with PCA, LDA
feature_selector_methods = [FeaturesSelector.NO_REDUCTION, FeaturesSelector.LDA, FeaturesSelector.PCA]
classification_methods = [(Classifier.LOGISTIC, lor_dict),
                          (Classifier.GAUSSIAN_NAIVE_BAYES, gnb_dict),
                          (Classifier.NEURAL_NETWORK, nn_dict),
                          (Classifier.LINEAR, linear_dict),
                          (Classifier.SVM, svm_dict),
                          (Classifier.LDA, lda_dict),
                          (Classifier.QDA, qda_dict)]

if not os.path.exists('results'):
    os.makedirs('results')

for cl_method in classification_methods:

    if cl_method[0] == Classifier.SVM:
        sets_original, class_names = load_data(scaler_kind=STD_SCALER)

    else:
        sets_original, class_names = load_data()

    if USE_CNN:
        feature_extractor = CNN()
        sets_original.train.x, sets_original.eval.x, sets_original.test.x = feature_extractor.extract(
            sets_original.train.x, sets_original.eval.x, sets_original.test.x)

    for fs_method in feature_selector_methods:

        number_of_features = [1024 if USE_CNN else 784]

        if fs_method == FeaturesSelector.PCA:
            number_of_features = range(10, 1024, 10) if USE_CNN else range(5, 785, 5)

        if fs_method == FeaturesSelector.LDA:
            number_of_features = range(1, 10)

        # Preparing the saving file
        log_file_name = 'results/' + cl_method[0] + '_' + fs_method + ('_cnn' if USE_CNN else '') + '.csv'
        print(log_file_name)

        with open(log_file_name, 'w') as log:
            # Creating the file and set the column names
            log.write("NumFeature;TrainingAccuracy;ValidationAccuracy\n")
            print("The file has been created!")

        accuracy_log = []

        for nf in number_of_features:

            accuracies = {'train': 0, 'eval': 0, 'test': 0}

            for _ in range(NUM_ATTEMPTS):

                sets = reshuffle(sets_original)

                classifier = Classifier(cl_method[0], **cl_method[1])
                selector = FeaturesSelector(fs_method, nf)
                sets = selector.fit(sets)

                train_predict, eval_predict = classifier.get_predictions(features=sets.train.x,
                                                                         labels=sets.train.y,
                                                                         eval_features=sets.eval.x,
                                                                         eval_labels=sets.eval.y)

                accuracies['train'] = accuracies['train'] + sum(
                    [train_predict[i] == sets.train.y[i] for i in range(len(train_predict))]) / len(train_predict)

                accuracies['eval'] = accuracies['eval'] + sum(
                    [eval_predict[i] == sets.eval.y[i] for i in range(len(eval_predict))]) / len(eval_predict)

            accuracies['train'] = accuracies['train'] / NUM_ATTEMPTS
            accuracies['eval'] = accuracies['eval'] / NUM_ATTEMPTS

            accuracy_log.append((nf, accuracies['train'], accuracies['eval'], accuracies['test']))

            with open(log_file_name, 'a') as log:
                log.write(
                    "{};{:.4};{:.4}\n".format(nf, accuracies['train'], accuracies['eval']))

        # Plot the chart of the data using accuracy_log
        nf_list = [int(el[0]) for el in accuracy_log]
        train_acc_list = [el[1] for el in accuracy_log]
        eval_acc_list = np.array([el[2] for el in accuracy_log])
        index = np.argmax(eval_acc_list)
        nf_max = nf_list[index]
        # test_acc_max = test_acc_list[index]
        eval_acc_max = eval_acc_list[index]

        plt.scatter(nf_list, train_acc_list, s=2, label="training accuracy")
        plt.scatter(nf_list, eval_acc_list, s=2, label="validation accuracy")
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(log_file_name[:-4] + '.png')
        plt.clf()

        # Calculation of the best found model in the whole training set = train_set + eval_set
        for _ in range(NUM_ATTEMPTS):

            sets = reshuffle(sets_original)

            classifier = Classifier(cl_method[0], **cl_method[1])
            selector = FeaturesSelector(fs_method, nf_max)
            sets = selector.fit(sets)

            train_predict, eval_predict, test_predict = classifier.get_predictions(features=sets.train.x,
                                                                                   labels=sets.train.y,
                                                                                   eval_features=sets.eval.x,
                                                                                   eval_labels=sets.eval.y,
                                                                                   test_features=sets.test.x)

            accuracies['test'] = accuracies['test'] + sum(
                [test_predict[i] == sets.test.y[i] for i in range(len(test_predict))]) / len(test_predict)

        with open(log_file_name, 'a') as log:
            log.write(
                "{};{};{:.4}\n".format(nf_max, 'test', accuracies['test'] / NUM_ATTEMPTS))
        print("The test accuracy is: ", accuracies['test'] / NUM_ATTEMPTS)
