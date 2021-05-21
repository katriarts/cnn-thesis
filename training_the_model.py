# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:42:36 2020

@author: Kat Ria
"""
import pandas as pd
import numpy as np

####################
#loading the datasets#
####################
basketball = pd.read_csv('csv_5000/bbasketball.csv')
baseball = pd.read_csv('csv_5000/bbaseball.csv')
shark = pd.read_csv('csv_5000/sshark.csv')
dolphin = pd.read_csv('csv_5000/ddolphin.csv')
duck = pd.read_csv('csv_5000/dduck.csv')
bird = pd.read_csv('csv_5000/bbird.csv')
van = pd.read_csv('csv_5000/vvan.csv')
ambulance = pd.read_csv('csv_5000/aambulance.csv')
knee = pd.read_csv('csv_5000/kknee.csv')
leg = pd.read_csv('csv_5000/lleg.csv')

####################
#add column names for each dataset#
####################
basketball = np.c_[basketball, np.zeros(len(basketball))]
baseball = np.c_[baseball, np.ones(len(baseball))]
shark = np.c_[shark, 2 * np.ones(len(shark))]
dolphin = np.c_[dolphin, 3 * np.ones(len(dolphin))]
duck = np.c_[duck, 4 * np.ones(len(duck))]
bird = np.c_[bird, 5 * np.ones(len(bird))]
van = np.c_[van, 6 * np.ones(len(van))]
ambulance = np.c_[ambulance, 7 * np.ones(len(ambulance))]
knee = np.c_[knee, 8 * np.ones(len(knee))]
leg = np.c_[leg, 9 * np.ones(len(leg))]

####################
#check 4x5 samples#
####################
import matplotlib.pyplot as plt
def plot_samples(input_array, rows=4, cols=5, title=''):
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)
    
    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

plot_samples(basketball, title='sample basketball drawings')
plot_samples(baseball, title='sample baseball drawings')
plot_samples(shark, title='sample shark drawings')
plot_samples(dolphin, title='sample dolphin drawings')
plot_samples(duck, title='sample duck drawings')
plot_samples(bird, title='sample bird drawings')
plot_samples(van, title='sample van drawings')
plot_samples(ambulance, title='sample ambulance drawings')
plot_samples(knee, title='sample knee drawings')
plot_samples(leg, title='sample leg drawings')

####################
#merge datasets; split features from labels; convert as float#
####################
X = np.concatenate((basketball[:5000,:-1], baseball[:5000,:-1], shark[:5000,:-1], dolphin[:5000,:-1], duck[:5000,:-1], bird[:5000,:-1], van[:5000,:-1], ambulance[:5000,:-1], knee[:5000,:-1], leg[:5000,:-1]), axis=0).astype('float32')
y = np.concatenate((basketball[:5000,-1], baseball[:5000,-1], shark[:5000,-1], dolphin[:5000,-1],duck[:5000,-1],bird[:5000,-1], van[:5000,-1], ambulance[:5000,-1], knee[:5000,-1], leg[:5000,-1]), axis=0).astype('float32') #the last column
#print(X.shape)
#print(y.shape)

####################
#train and test split; divide by 255 for normalization(0-1 values)#
####################
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X/255.0, y, test_size=0.2, random_state=0)

####################
####################
#CNN#
####################
####################

####################
#one-hot encoding the outputs#
####################
from keras.utils import np_utils

y_train_cnn = np_utils.to_categorical(y_train)#.astype('float32')
y_test_cnn = np_utils.to_categorical(y_test)#.astype('float32')
num_classes = y_test_cnn.shape[1]

####################
#reshape to be [samples][pixels][width][height]#
####################
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')


####################
#CNN model#
####################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def cnn_structure():
    model = Sequential()
    #Conv2D:learned a total of 32 filters, add first layer
    model.add(Conv2D(16,(3,3), activation='relu', input_shape=(28, 28,1)))
                                                                                                                #maxpooling2d: used to reduce the spatial dimensions of the output volume
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #add 2nd layer
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #add a dropout layer
    model.add(Dropout(0.2)) #20% dropout
    #add a flattening layer
    model.add(Flatten()) #reduce dimensionality to a linear array
    
    #add a layer with 250 neurons
    model.add(Dense(250, activation='relu'))    
    #add a layer with 50 neurons
    model.add(Dense(50, activation='relu'))   
    #add a layer, softmax activation
    model.add(Dense(num_classes, activation='softmax'))
    
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
 
    return model


#np.random.seed(0)
cnn_model = cnn_structure()
cnn_model.summary()    

####################
#fit cnn model#
####################
#note: train samples and validate samples have different samplle values due to some neurons being dropped (in the model)
history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=40, batch_size=16, validation_data=(X_test_cnn, y_test_cnn))

####################
#save model so no need to repeatedly run the cnn_model.fit#
####################
#cnn_model.save('cnn_model.model')
cnn_model.save('cnn_model.h5')

#import pickle
#with open('cnn_model.pkl', 'wb') as file:
 #   pickle.dump(cnn_model, file)

####################
#summarize values#
####################
acc= history.history['accuracy']
loss = history.history['loss']
vacc = history.history['val_accuracy']
vloss = history.history['val_loss']
print()
print( "Train Accuracy: ",np.array(acc).mean()*100,'%')
print("Train Loss: ", np.array(loss).mean()*100,'%')
print("Validation Accuracy: ", np.array(vacc).mean()*100,'%')
print("Validation Loss: ", np.array(vloss).mean()*100,'%')
print()

vacc2 = np.array(history.history['val_accuracy']).mean()
print('Model Accuracy: ', np.mean(vacc2)*100, "(+/-", np.std(vacc2)*100,")")


####################
#class names#
####################   
class_names = ['basketball', 'baseball', 'shark', 'dolphin', 'duck', 'bird', 'van', 'ambulance', 'knee', 'leg']


####################
#confusion matrix#
####################
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred_cnn = cnn_model.predict_classes(X_test_cnn, verbose=2)
cm = confusion_matrix(y_test, y_pred_cnn)

def confusion_matrix(confusion_matrix, class_names, figsize = (10, 7), font_size=14):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    plt.figure(figsize = figsize)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt = "d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')#, fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')#, fontsize=fontsize)

    plt.ylabel('true label')
    plt.xlabel('predicted label')


confmat = confusion_matrix(cm, class_names, figsize=(10, 7))#, fontsize=14)

####################
#misclassification when y_pred and y_test are different#
####################
misclassified = X_test[y_pred_cnn != y_test]
plot_samples(misclassified, rows=5, cols=10, title='misclassification')

#% of the images were predicted with _% cetainty
cnn_probab = cnn_model.predict(X_test_cnn, batch_size=10, verbose=2)
#extract the probability for the label that was predicted:
p_max = np.amax(cnn_probab, axis=1)
plt.hist(p_max, normed=True, bins=list(np.linspace(0,1,11)))
plt.xlabel('p of predicted class')


#visualize accuracy
plt.subplot(212)
plt.plot(acc, label='train')
plt.plot(vacc, label='validation')
plt.title("model accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()

#visualize loss
plt.subplot(212)
plt.plot(loss, label='train')
plt.plot(vloss, label='validation')
plt.title("model loss")
plt.legend()
plt.show()

####################
#get accuracy, precision, recall, f1 score and cohens kappa#
####################  
########
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

#predict xtest probability
yhat_probs = cnn_model.predict(X_test_cnn, verbose=0)
#predict xtest 
yhat_classes = cnn_model.predict_classes(X_test_cnn, verbose=0).reshape(-1,1)

yhat_inverse = np.argmax(yhat_probs, axis=1).reshape(-1,1)
ytest_inverse = np.argmax(y_test_cnn, axis=1).reshape(-1,1)

#reduce to 1d array
yhat_probs = yhat_inverse.flatten()#[:,0]
yhat_classes = yhat_classes.flatten()#[:,0]

#accuracy
accuracy = accuracy_score(ytest_inverse, yhat_classes)
print('Accuracy: %f' % accuracy)
#precision
precision = precision_score(ytest_inverse, yhat_classes, average = 'weighted')
print('Precision: %f' % precision)
#recall
recall = recall_score(ytest_inverse, yhat_classes, average = 'weighted')
print('Recall: %f' % recall)
#f1 score
f1 = f1_score(ytest_inverse, yhat_classes, average = 'weighted')
print('F1 score: %f' % f1)

#kappa
kappa = cohen_kappa_score(ytest_inverse, yhat_classes)
print('Cohens Kappa: %f' % kappa)


####################
#read image test#
####################   
from tensorflow.keras import models
model = models.load_model('cnn_model.h5')

import matplotlib.pyplot as plt
from random import randint

idx = randint(0, len(X_test_cnn))
img = X_test_cnn[idx]
plt.imshow(img.squeeze())

pred = model.predict(np.expand_dims(img, axis=0))[0]
ind = (-pred).argsort()[:3]
latex = [class_names[x] for x in ind]
print(latex)
####################
#!!!!!!!!!!!!!!!!!!#
####################  


