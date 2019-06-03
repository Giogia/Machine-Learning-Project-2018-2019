from FeaturesSelector import FeaturesSelector
from Classifier import Classifier
from DataHandler import load_data, STD_SCALER
from time import time
import os
from CNN import CNN
from matplotlib import pyplot as plt
import numpy as np

################################################################################
################################## PARAMETERS ##################################
################################################################################

# Number of attempts that have to be averaged
NUM_ATTEMPTS = 1
USE_CNN = True

# Preparing the files where to redirect the standard error and the standard output
# sys.stdout = open('out.log', 'w')
# sys.stderr = open('err.log', 'w')

# The default configuration of the parameters for the logistic regression
lor_dict = {'penalty': 'l2', 'dual': False, 'tol': 1e-4, 'C': 0.5, 'fit_intercept': True, 'intercept_scaling': 1,
            'class_weight': None, 'random_state': None, 'solver': 'lbfgs', 'max_iter': 700, 'multi_class': 'auto',
            'verbose': 0, 'warm_start': False, 'n_jobs': 1}

# The parameters of the logistic regression that are modified from the default value

# The default configuration of the parameters for the svm
svm_dict = {
    'C': 1.0,  # Penalty parameter C of the error term.
    'kernel': 'rbf',
    # Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’,
    # ‘sigmoid’, ‘precomputed’
    'degree': 2,  # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    'gamma': 'auto',  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    'coef0': 0.0,  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    'shrinking': True,  # Whether to use the shrinking heuristic.
    'probability': False,  # Whether to enable probability estimates
    'tol': 0.001,  # Tolerance for stopping criterion.
    'cache_size': 200,  # Specify the size of the kernel cache (in MB)
    'class_weight': None,  # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are
    # supposed to have weight one
    'verbose': False,  # Enable verbose output
    'max_iter': -1,  # Hard limit on iterations within solver, or -1 for no limit.
    'decision_function_shape': 'ovr',  # Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples,
    # n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision
    # function
    'random_state': None,  # The seed of the pseudo random number generator used when shuffling the data for probability
    # estimates
}

# The default configuration of the parameters for the gaussian naive bayes
gnb_dict = {
    'priors': None,  # Array of dimension equal to the number of classes.
    # It contains the prior distributions of the classes.
    'var_smoothing': 1e-9  # Do not touch :)
}

# The default configuration of the parameters for the linear discriminant analysis
lda_dict = {
    'solver': 'lsqr',
    'shrinkage': None,
    'priors': None,
    'n_components': None,
    'store_covariance': False,
    'tol': 0.0001,
}

# The default configuration of the parameters for the neural network
nn_dict = {'batch_size': 128,
           'epochs': 150,
           'verbose': 0,
           'optimizer': 'adam',
           'loss': 'sparse_categorical_crossentropy',
           'metrics': ['accuracy']}

# If more methods are added, let's add it here
# feature_selector_methods = [FeaturesSelector.NO_REDUCTION, FeaturesSelector.PCA, FeaturesSelector.LDA]
feature_selector_methods = [FeaturesSelector.NO_REDUCTION]
# classification_methods = [(Classifier.LOGISTIC, lor_dict), (Classifier.GAUSSIAN_NAIVE_BAYES, gnb_dict),
# (Classifier.SVM, svm_dict), (Classifier.NEURAL_NETWORK, nn_dict)]
classification_methods = [(Classifier.NEURAL_NETWORK, nn_dict)]
# classification_methods = [(Classifier.LDA, lda_dict)]

################################################################################
#################################### SCRIPT ####################################
################################################################################

if not os.path.exists('results'):
    os.makedirs('results')

for cl_method in classification_methods:
    for fs_method in feature_selector_methods:

        number_of_features = [12800 if USE_CNN else 784]

        if fs_method == FeaturesSelector.PCA:
            number_of_features = range(50, 12800 if USE_CNN else 785, 5)

        if fs_method == FeaturesSelector.LDA:
            number_of_features = range(1, 10)

        # Preparing the saving file
        log_file_name = 'results/' + cl_method[0] + '_' + fs_method + '_' + str(time()).split('.')[0] + '.csv'

        with open(log_file_name, 'w') as log:
            # Creating the file and set the column names
            log.write("NumFeature;TrainingAccuracy;ValidationAccuracy\n")
            print("The file has been created!")

        accuracy_log = []

        for nf in number_of_features:

            accuracies = {'train': 0, 'eval': 0, 'test': 0}

            for _ in range(NUM_ATTEMPTS):

                if cl_method[0] == Classifier.SVM:
                    sets, class_names = load_data(scaler_kind=STD_SCALER)

                else:
                    sets, class_names = load_data()

                if USE_CNN:
                    feature_extractor = CNN()
                    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x,
                                                                                       sets.test.x)
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
                print("Attempt finished!")

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

        # Leave a space for description
        plt.xlabel("\n")
        plt.figtext(0.5, 0.05, "Best Number of Features = {}".format(nf_max if nf_max != 0 else 784)
                    + "   Best Evaluation Accuracy = {}".format(eval_acc_max),
                    wrap=True, horizontalalignment='center', fontsize=10)

        plt.grid(True)
        plt.legend(loc='best')
        plt.title(log_file_name[8:-4])
        plt.tight_layout()
        plt.savefig(log_file_name[:-4] + '.png')
        plt.clf()

        # Calculation of the best found model in the whole training set = train_set + eval_set
        for _ in range(NUM_ATTEMPTS):

            if cl_method == Classifier.SVM:
                sets, class_names = load_data(scaler_kind=STD_SCALER)

            else:
                sets, class_names = load_data()

            if USE_CNN:
                feature_extractor = CNN()
                sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x,
                                                                                   sets.test.x)

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
            print("Attempt finished!")
        print("The test accuracy is: ", accuracies['test'] / NUM_ATTEMPTS)
