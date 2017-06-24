import utils
import model
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

X, X_test, Y, Y_test = utils.returner()


sess = tf.InteractiveSession()
cnnModel = model.Model()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

def train(training_data, training_label, test_data, test_label):


    batch_count = 0
    for i in range(3500):

        if batch_count == 14900:
            batch_count = 14899
        
        if i%100 == 0:
            train_accuracy = cnnModel.accuracy.eval(feed_dict={cnnModel.x:training_data[batch_count:batch_count+100],cnnModel.y_:training_label[batch_count:batch_count+100]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        cnnModel.train_step.run(feed_dict={cnnModel.x: training_data[batch_count:batch_count+100],cnnModel.y_: training_label[batch_count:batch_count+100]})
        batch_count = batch_count + 100
        if batch_count >= 14990:
            batch_count = 0
       


    save_path = "Path to save your model"
    saver.save(sess, save_path)

    accuracy_batch = 0
    accuracy_keep = 0
    for j in range(49):


        accuracy_keep += cnnModel.accuracy.eval(feed_dict={cnnModel.x:test_data[accuracy_batch:accuracy_batch+100], cnnModel.y_:test_label[accuracy_batch:accuracy_batch+100]})
        print("The total accuracy so far is :",accuracy_keep/((j+1)))

        accuracy_batch += 100
        
    


def verification():
    batch_count = 0
    accuracy_test = 0
    for i in range(49):
        accuracy_test += cnnModel.accuracy.eval(feed_dict={cnnModel.x:X_test[batch_count:batch_count+100], cnnModel.y_:Y_test[batch_count:batch_count+100]})
        # print(accuracy_test/((i+1)))
        batch_count += 100
        
    print("The accuracy of the test set is : ",accuracy_test/49)

kf = KFold(n_splits=4)
kf.get_n_splits(X)


for train_index, test_index in kf.split(X):
    
    X_train = np.zeros((15000, 128, 128), dtype='float64')
    X_test = np.zeros((5000,128,128), dtype='float64')
    Y_train = np.zeros((15000))
    Y_test = np.zeros((5000))
    
    print(train_index, test_index)
    
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    print("X_Train shape :", X_train.shape, "X_test shape :", X_test.shape, "Y_train shape :", Y_train.shape, "Y_test shape :", Y_test.shape)
    
    one_hot_vectorTrain = np.zeros((15000,2), dtype=np.float32)
    one_hot_vectorTest = np.zeros((5000,2), dtype=np.float32)
    
    for i in range(Y_train.shape[0]):
        one_hot_vectorTrain[i][int(Y_train[i])] = 1.0

    for j in range(Y_test.shape[0]):
        one_hot_vectorTest[j][int(Y_test[j])] = 1.0
    
    Y_train = one_hot_vectorTrain
    Y_test = one_hot_vectorTest
    
    train(X_train,Y_train,X_test,Y_test)
    verification()