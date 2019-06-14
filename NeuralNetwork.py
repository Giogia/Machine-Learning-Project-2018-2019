from src.CNN_v1 import CNN
from src.Classifier import Classifier
from src.DataHandler import load_data
from src.FeaturesSelector import FeaturesSelector

'''
The default dictionary for the neural network parameters

nn_dict = {
    'batch_size': 128,                          # number of images accepted by the newtork
    'epochs': 15,                               # computation iterations 
    'verbose': 0,                               # 1 to log neural network information
    'optimizer': 'adam',                        # String (name of optimizer) or optimizer instance, See tf.keras.optimizers
    'loss': 'sparse_categorical_crossentropy',  # String (name of objective function) or objective function, See tf.losses
    'metrics': ['accuracy']                     # List of metrics to be evaluated by the model during training and testing
}

'''

nn_dict = {
    'batch_size': 128,
    'epochs': 15,
    'verbose': 0,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}


USE_CNN = True
OPTIMAL_FEATURE_NUMBER_PCA = 370 if not USE_CNN else 1000

sets, class_names = load_data()

if USE_CNN:

    # Compute high level features
    feature_extractor = CNN()
    sets.train.x, sets.eval.x, sets.test.x = feature_extractor.extract(sets.train.x, sets.eval.x, sets.test.x)

nn_classifier = Classifier(Classifier.NEURAL_NETWORK, **nn_dict)

feature_selector = FeaturesSelector(FeaturesSelector.PCA, OPTIMAL_FEATURE_NUMBER_PCA)
sets = feature_selector.fit(sets)

train_predict, eval_predict, test_predict = nn_classifier.get_predictions(features=sets.train.x,
                                                                           labels=sets.train.y,
                                                                           eval_features=sets.eval.x,
                                                                           eval_labels=sets.eval.y,
                                                                           test_features=sets.test.x)

train_accuracy = sum([train_predict[i] == sets.train.y[i] for i in range(len(train_predict))]) / len(train_predict)
eval_accuracy = sum([eval_predict[i] == sets.eval.y[i] for i in range(len(eval_predict))]) / len(eval_predict)
test_accuracy = sum([test_predict[i] == sets.test.y[i] for i in range(len(test_predict))]) / len(test_predict)

print("\n\n\nTrain Accuracy: {}".format(train_accuracy))
print("Validation Accuracy: {}".format(eval_accuracy))
print("Test Accuracy: {}".format(test_accuracy))
