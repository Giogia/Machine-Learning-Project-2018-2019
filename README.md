## <div align="center">Machine Learning Project</div>
#  <div align="center">Classifying Clothes using Fashion MNIST dataset</div>

Our project aims to compare different ML techniques with respect to the final accuracy obtained by the classifiers on F-MNIST dataset. Moreover, we will use the best algorithm in order to perform the classification of real images from the web.


When executing the code is possible to choose between two alternatives: running NumberFeatureExploration that will start the procedure of training on all the selected classifiers and feature extraction method or simply run the specific method of interest with the selected feature extraction method.

---

#### NumberFeatureExploration.py

This file is responsible for testing the various classification methods with the various possible combinations of feature extraction methods and convolution, for a defined number of times.

Change the following line to change the total number of attempts to execute for each test, the retrieved accuracy is given by the average of the results.
```python
NUM_ATTEMPS = 5
```

If it's not desired to use the convolutional neural network to extract new features. change this variable to False, if it's left to True, the method will use a pre-trained model to extract the features.
```python
USE_CNN = True
```

In order to select which classification method and which feature extraction method change the content of the following lists
```python
feature_selector_methods = [FeaturesSelector.NO_REDUCTION, FeaturesSelector.LDA, FeaturesSelector.PCA]
classification_methods = [(Classifier.LOGISTIC, lor_dict),
                          (Classifier.GAUSSIAN_NAIVE_BAYES, gnb_dict),
                          (Classifier.NEURAL_NETWORK, nn_dict),
                          (Classifier.LINEAR, linear_dict),
                          (Classifier.SVM, svm_dict),
                          (Classifier.LDA, lda_dict),
                          (Classifier.QDA, qda_dict)]
```

Lastly, each classification method can be furtherly personalized by changing the associated dictionary of values; given that the classification methods are implemented using Scikit-Learn library, the accepted keywords are the only one accepted by the library.
Here we can see an example of the dictionary for linear regression
```python
linear_dict = {
    'fit_intercept': True,
    'normalize': False,
    'copy_X': True,
    'n_jobs': None
}
```

#### Classification methods files

Apart from NumberFeatureExplorer.py is possible to use also the proprietary file for each classification method, in this case only one feature extraction method will be used, by default the values are set to the configuration that during our tests delivered the best performance.
In order to change the number of features used or change the feature extraction method alter the following lines
```python
OPTIMAL_FEATURE_NUMBER_LDA = 9

feature_selector = FeaturesSelector(FeaturesSelector.LDA, OPTIMAL_FEATURE_NUMBER_LDA)
```
or alternatively
```python
OPTIMAL_FEATURE_NUMBER_PCA = 415 if not USE_CNN else 920

feature_selector = FeaturesSelector(FeaturesSelector.PCA, OPTIMAL_FEATURE_NUMBER_PCA)
```

#### CNN_training.ipynb

This file is responsible for the training of the CNN if it's necessary to modify or change the trained model, it's possible to change this code and re-run it. In the end, a new model will be generated.

#### Machine Learning Application.ipynb

This is the file that was used during the creation of the video-demo; if desired it's possible to run it in order to prove the correctness of the described results.
For the demo 3 custom datasets of 5 images each were provided, the datasets present different difficulty and the one on which is desired to execute the code can be retrieved by changing the value of CHOICE as described
```python
CHOICE = 0 # 0=standard, 1=advanced, 2=hard
```
