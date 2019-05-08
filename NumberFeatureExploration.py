from FeaturesSelector import FeaturesSelector
from Classifier import Classifier
from DataHandler import load_data
from time import time
from CNN import CNN
from matplotlib import pyplot as plt
import numpy as np
import sys

################################################################################
################################## PARAMETERS ##################################
################################################################################

# Number of attempts that have to be averaged
NUM_ATTEMPTS = 5
USE_CNN = False

# Preparing the files where to redirect the standard error and the standard output
# sys.stdout = open('out.log', 'w')
# sys.stderr = open('err.log', 'w')

# The default configuration of the parameters for the logistic regression
lor_dict = {
    'penalty': 'l2',        # 'l1' or 'l2'
    'dual': False,          # True if #feature > #samples (only if l2 active)
    'tol': 1e-4,            # tollerance for early stopping
    'C': 1.0,               # inverse 8of the regularization term of LoR smaller values implies stronger regularization
    'fit_intercept': True,  # True if we want the Bias
    'intercept_scaling': 1, # Useful only when solver 'liblinear'.
                            # The higher it is, the less the bias are regularized, the bigger they can become
    'class_weight': None,   # None,'balanced' or dict. For giving a weight to the various classes.
                            # Usefull if unbalanced datsets.
    'random_state': None,   # Seed for initializing the random generator --> for experiments reproducibility
    'solver': 'warn',       # 'newton-cg','lbfgs','liblinear','sag','saga'.
                            # 'liblinear' good for small datasets.
                            # 'sag' and 'saga' are faster on big datasets.
                            # 'liblinear' has a one-vs-rest approach, the others use multinnomials.
                            # 'newton-cg', 'lbfgs' and 'sag' use just L2 regularization.
                            # 'sag' and 'saga' guarantee a fast convergence if the features are approximated on the same scale.
    'max_iter': 100,        # maximum number of iterations
    'multi_class': 'auto',  # 'ovr', 'multinomial' or 'auto'. We should use multinomial.
                            # If we use liblinear solver, we have to use ovr.
    'verbose': 1,           # 0,1 o 2. Levels of verbosity.
    'warm_start': False,    # It True it reuse the solution of previous fit.
    'n_jobs': None          # Number of processors used by the computation.
}

# The parameters of the logistic regression that are modified from the default value
lor_dict['max_iter'] = 700
lor_dict['verbose'] = 0
lor_dict['C'] = 0.5
lor_dict['solver'] = 'lbfgs'
lor_dict['n_jobs'] = 1

# The default configuration of the parameters for the gaussian naive bayes
gnb_dict = {
    'priors': None,  # Array of dimension equal to the number of classes.
    # It contains the prior distributions of the classes.
    'var_smoothing': 1e-9  # Do not touch :)
}

# If more methods are added, let's add it here
# feature_selector_methods = [FeaturesSelector.NO_REDUCTION, FeaturesSelector.PCA, FeaturesSelector.LDA]
feature_selector_methods = [FeaturesSelector.LDA]
# classification_methods = [(Classifier.LOGISTIC,lor_dict), (Classifier.GAUSSIAN_NAIVE_BAYES,gnb_dict)]
classification_methods = [(Classifier.GAUSSIAN_NAIVE_BAYES, gnb_dict)]


################################################################################
#################################### SCRIPT ####################################
################################################################################

for cl_method in classification_methods:
    for fs_method in feature_selector_methods:

        number_of_features = [0]


        if fs_method == FeaturesSelector.PCA:
            number_of_features = range(5, 1024 if USE_CNN else 785, 5)
            # number_of_features = range(5,10,5)

        if fs_method == FeaturesSelector.LDA:
            number_of_features = range(1, 10)
            # number_of_features = range(1,2)

        # Preparing the saving file
        log_file_name = 'results/' + cl_method[0] + '_' + fs_method + '_' + str(time()).split('.')[0] + '.csv'

        with open(log_file_name, 'w') as log:
            # Creating the file and set the column names
            log.write("NumFeature;TrainingAccuracy;ValidationAccuracy;TestAccuracy\n")

        accuracy_log = []

        for nf in number_of_features:

            # print("Method: {} \tNumber Feature: {}".format(fs_method, nf))

            accuracies = {'train': 0, 'eval': 0, 'test': 0}

            for _ in range(NUM_ATTEMPTS):

                sets, class_names = load_data(linearized=True)
                if USE_CNN:
                    feature_extractor = CNN()
                    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)
                classifier = Classifier(cl_method[0], **cl_method[1])
                selector = FeaturesSelector(fs_method, nf)
                sets = selector.fit(sets)
                train_predict, eval_predict, test_predict = classifier.get_predictions(features=sets.train.x,
                                                                                       labels=sets.train.y,
                                                                                       eval_features=sets.eval.x,
                                                                                       eval_labels=sets.eval.y,
                                                                                       test_features=sets.test.x)

                accuracies['train'] = accuracies['train'] + sum(
                    [train_predict[i] == sets.train.y[i] for i in range(len(train_predict))]) / len(train_predict)

                accuracies['eval'] = accuracies['eval'] + sum(
                    [eval_predict[i] == sets.eval.y[i] for i in range(len(eval_predict))]) / len(eval_predict)

                accuracies['test'] = accuracies['test'] + sum(
                    [test_predict[i] == sets.test.y[i] for i in range(len(test_predict))]) / len(test_predict)

            accuracies['train'] = accuracies['train'] / NUM_ATTEMPTS
            accuracies['eval'] = accuracies['eval'] / NUM_ATTEMPTS
            accuracies['test'] = accuracies['test'] / NUM_ATTEMPTS

            accuracy_log.append((nf,accuracies['train'],accuracies['eval'],accuracies['test']))

            with open(log_file_name, 'a') as log:
                log.write(
                    "{};{:.4};{:.4};{:.4}\n".format(sets.train.x.shape[1], accuracies['train'], accuracies['eval'],
                                                    accuracies['test']))

        # Plot the chart of the data using accuracy_log
        nf_list         = [el[0] for el in accuracy_log]
        train_acc_list  = [el[1] for el in accuracy_log]
        eval_acc_list   = [el[2] for el in accuracy_log]
        test_acc_list   = [el[3] for el in accuracy_log]

        index = np.argmax(test_acc_list)
        nf_max = nf_list[index]
        test_acc_max = test_acc_list[index]



        plt.scatter(nf_list,train_acc_list,label="training accuracy")
        plt.scatter(nf_list,eval_acc_list,label="validation accuracy")
        plt.scatter(nf_list,test_acc_list,label="test accuracy")
        plt.annotate("Best Test Accuracy = {}".format(test_acc_max), xy=(nf_max, test_acc_max), xytext=(nf_max, test_acc_max -0.1),arrowprops=dict(facecolor='black', shrink=0.005),)
        plt.legend(loc='best')
        plt.title(log_file_name[8:-4])
        plt.tight_layout()
        plt.savefig(log_file_name[:-4]+'.png')