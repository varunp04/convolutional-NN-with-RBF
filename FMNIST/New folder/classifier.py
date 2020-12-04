import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers 
import tensorflow as tf
import math
from tensorflow.keras.utils import to_categorical

def update_dictionary_items(dict1, dict2):
    
    if dict2 is None:
        return dict1
    for k in dict1:
        if k in dict2:
            dict1[k] = dict2[k]

    return dict1

def get_d(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum = sum+((x1[i] - x2[i]) ** 2) 
    
    return math.sqrt(sum)    

class convnet():
    def __init__(self, parameters={}):
        self.params = update_dictionary_items({'k':10, 'epochs': 10, 'batch_size': 128, 'act':'relu','filter1': 64, 'filter2': 128, 'filter3': 128, 'filter4': 128}, parameters)
        self.weights = None
    def modelcons(self):
    #construction of model
        self.bconv = models.Sequential()
        self.bconv.add(layers.Conv2D(self.params['filter1'],(3,3),activation = self.params['act'],input_shape=(28,28,1)))
        self.bconv.add(layers.MaxPooling2D((2,2)))
        self.bconv.add(layers.Conv2D(self.params['filter2'],(3,3),activation = self.params['act']))
        self.bconv.add(layers.MaxPooling2D((2,2)))
        self.bconv.add(layers.Conv2D(self.params['filter3'],(3,3),activation = self.params['act']))
        self.bconv.add(layers.Flatten())

    def learn(self, X, y):
        self.net = models.Sequential()
        self.net.add(self.bconv)
        self.net.add(layers.Dense(self.params['filter4'],activation=self.params['act']))
        self.net.add(layers.Dense(10,activation ='softmax'))
        y = to_categorical(y)
        #compile
        self.net.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

        #training
        self.net.fit(X,y,epochs=self.params['epochs'],batch_size=self.params['batch_size'])

    def predict(self, Xtest,ytest):
        test_loss,test_acc = self.net.evaluate(Xtest,ytest)
        return test_acc

    

    def rbfn_test(self, Xtest):
        RBF_X_test = self.rbf_list(Xtest, self.cluster_centers, self.std_list)
        pred = RBF_X_test.dot(self.weights)
        pred = tf.nn.softmax( pred,axis=1)
        print(pred[0])
        print(pred[1])
        arr = np.argmax(pred,axis =1)
        arr = np.ravel(arr)
        print(arr)

        return arr


    def baseconv(self,sample_count,X,y):
        conv_base = self.bconv
        features = np.zeros(shape=(sample_count,1152))
        labels = np.zeros(shape=(sample_count))
        pre_val = conv_base.predict(X)
        i=0
        while(i<sample_count):    
            features[i] = pre_val[i]
            labels[i] = y[i]
            i = i + 1 
        return features,labels

    def convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def rbfn(self,X,y):
        print(self.params['k'])
        kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=self.params['k'],distance_metric='squared_euclidean',use_mini_batch=True)
        def input_fn():
            return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(X, dtype=tf.float32), num_epochs=1)
        kmeans.train(input_fn)
        y = self.convert_to_one_hot(y,10)

        self.cluster_centers = kmeans.cluster_centers()
        dMax = np.max([get_d(c1, c2) for c1 in self.cluster_centers for c2 in self.cluster_centers])
        self.std_list = np.repeat(dMax / np.sqrt(2 * self.params['k']), self.params['k'])
        print('dmax', dMax)
        self.std_list =  np.reshape(self.std_list,(self.params['k'],1))
        
        RBF_X = self.rbf_list(X, self.cluster_centers, self.std_list)
        numsamples = X.shape[0]
        w = np.random.rand(self.params['k'],10)
        p = np.identity(self.params['k'])
        for i in range(numsamples):
            alpha = y[i,:] - RBF_X[i,:].dot(w)
            nm = p*(RBF_X[i,:].dot(RBF_X[i,:].T))
            din = RBF_X[i,:].dot(p.dot(RBF_X[i,:].T))
            p = p - ((nm.dot(p))/(1+din))
            g = p.dot(RBF_X[i].T)

            alpha = np.reshape(alpha,(10,1))
            g=np.reshape(g,(self.params['k'],1))
            w = w + g.dot(alpha.T)




        self.weights = w
        print(self.weights)
        

    def rbf(self, x, c, s):
        distance = get_d(x, c)
        
        return 1 / np.exp(-distance / s ** 2)

    def rbf_list(self, X, cs, sl):

        RBF_list = np.zeros((X.shape[0],self.params['k']))
        i=0
        for x in X:
            RBF_list[i] = ([self.rbf(x, c, s) for (c, s) in zip(cs, sl)])

            i=i+1
            if(i%1000 == 0):
                print(i)
        return np.array(RBF_list)

