import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
import os 
from PIL import Image
import numpy as np
ellipse = 'Galaxies/Galaxy_Color_Images/Ellipse_Images/'
spiral = 'Galaxies/Galaxy_Color_Images/Spiral_Images/'
path_sp = os.listdir(spiral)
path_el = os.listdir(ellipse)

##### Building a Training Data
el_train = []
for i in range(1, 10000):
    im = Image.open(ellipse + path_el[i])
    el_train.append(np.ravel(im.getdata()))
sp_train = [] 
for i in range(1, 10000):
    im = Image.open(spiral + path_sp[i])
    sp_train.append(np.ravel(im.getdata()))
## Creating a training set
y_el_train = []
y_sp_train = []
for i in range(0,len(el_train)):
    y_el_train.append(0)
for i in range(0,len(sp_train)):
    y_sp_train.append(1)
X_train = el_train + sp_train
y_train = y_el_train + y_sp_train

#### Creating Test set ######
el_test = []
for i in range(10001, 12500):
    im = Image.open(ellipse + path_el[i])
    el_test.append(np.ravel(im.getdata()))

sp_test = []
for i in range(10001, 12500):
     im = Image.open(spiral + path_sp[i])
     sp_test.append(np.ravel(im.getdata()))

X_test = el_test + sp_test
y_true = []
for i in range(len(el_test)):
    y_true.append(0)
for i in range(len(sp_test)):
    y_true.append(1)

n_components = [2, 5, 7, 10, 15, 20]

X_train = np.array(X_train)

print (' Now calculating metrics.......')

for n_component in range(0,len(n_components)):
    print '==============================================='
    print "When n_compenents ================ " + str(n_components[n_component])
    print '==============================================='
    pca = PCA(n_components=n_components[n_component], svd_solver='randomized', whiten=True).fit(X_train)
    #print('The Variance Ratios ')
    #print(pca.explained_variance_ratio_)
    #print(' The Variance ')
    #print(pca.explained_variance_)


    #print("Projecting the input data on the eigenfaces orthonormal basis")
    print('Applying PCA to the Training and Test Data')
    el_pca_test = pca.transform(el_test)
    sp_pca_test = pca.transform(sp_test)
    X_train_pca = pca.transform(X_train)
### Change here
    gamma = [0.01,0.05 ,0.07 ,0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    for i in range(0, len(gamma)):
        print("For gamma =========" + str(gamma[i]))
        print('************************************')

        print("Fitting the classifier to the training set")
    #rbf_svc = svm.SVC(kernel = 'rbf', class_weight = 'balanced', probability = True, gamma = 0.01)
        rbf_svc = svm.SVC(kernel = 'rbf', class_weight = 'balanced', probability = True, gamma = gamma[i])

        rbf_svc.fit(X_train_pca, y_train)
        print("Best estimator found by grid search:")

        pos_el = 0 
        el_predictions = []
        for i in range(0,len(el_pca_test)):
            el_predictions.append(rbf_svc.predict(el_pca_test[i].reshape(1,-1)))
        for i in range(0,len(el_pca_test)):
            if el_predictions[i] == [0]:
                pos_el = pos_el + 1
        print 'Are ellipses ' +  str(pos_el)
        print 'Number of Elliptical Tests ' + str(len(el_predictions)) + ' Accuracy: ' + str(pos_el/float(len(el_predictions))*100)

        pos_sp = 0 
        sp_predictions = []
        for i in range(0,len(sp_pca_test)):
            sp_predictions.append(rbf_svc.predict(sp_pca_test[i].reshape(1,-1)))
        for i in range(0,len(sp_pca_test)):
            if sp_predictions[i] == [1]:
                pos_sp = pos_sp + 1

        y_pred = el_predictions + sp_predictions
        print 'Are Sprials ' +  str(pos_sp) + ' Accuracy: ' + str(pos_sp/float(len(sp_predictions))*100)
        print 'Number of Spiral Tests ' + str(len(sp_predictions))
        print('**************************************')
        print('The Overall Accuracy ---------------------------------------')
        print(accuracy_score(y_true, y_pred))
        del y_pred

