from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import classifier as algs
from mlxtend.data import loadlocal_mnist



def getaccuracy(ytest, pred):
    
    TP =0

    for i in range(len(ytest)):
        if(ytest[i] == pred[i]):
            TP = TP + 1

    print(TP)



    acc = (TP/10000)
    print(acc)
    return acc

    
def g_accuracy(test,pred):
    TP = np.zeros(10)
    tot = np.zeros(10)

    for i in range(test.shape[0]):
        if(test[i] == pred[i]):
            TP[test[i]] = TP[test[i]] + 1

    for i in range(test.shape[0]):
        tot[test[i]] += 1


    acc = np.divide(TP,tot)
    print(acc)
    return np.prod(acc)

def s_cross_validate(K, x, y, Algorithm, parameters):
    all_acc = np.zeros((len(parameters), K))
    all_g = np.zeros((len(parameters), K))
    num_val_sam =len(x) // K

    xblock0 = np.zeros((num_val_sam,1152)); xblock2 = np.zeros((num_val_sam,1152)); 
    xblock3 = np.zeros((num_val_sam,1152)); xblock4 = np.zeros((num_val_sam,1152));
    xblock5 = np.zeros((num_val_sam,1152)); xblock6 = np.zeros((num_val_sam,1152)); 
    xblock7 = np.zeros((num_val_sam,1152)); xblock8 = np.zeros((num_val_sam,1152));
    xblock9 = np.zeros((num_val_sam,1152)); xblock1 = np.zeros((num_val_sam,1152));

    yblock0 = np.zeros(num_val_sam); yblock2 = np.zeros(num_val_sam); 
    yblock3 = np.zeros(num_val_sam); yblock4 = np.zeros(num_val_sam);
    yblock5 = np.zeros(num_val_sam); yblock6 = np.zeros(num_val_sam); 
    yblock7 = np.zeros(num_val_sam); yblock8 = np.zeros(num_val_sam);
    yblock9 = np.zeros(num_val_sam); yblock1 = np.zeros(num_val_sam);

    xblock0=xblock0.astype('float32')/255;xblock1=xblock1.astype('float32')/255;
    xblock2=xblock2.astype('float32')/255;xblock3=xblock3.astype('float32')/255;
    xblock4=xblock4.astype('float32')/255;xblock5=xblock5.astype('float32')/255;
    xblock6=xblock6.astype('float32')/255;xblock7=xblock7.astype('float32')/255;
    xblock8=xblock8.astype('float32')/255;xblock9=xblock9.astype('float32')/255;
    arrx = np.zeros(10)   ##block counter
    countx = np.zeros(10)     ##6000 counter

    x0=0;x1=0;x2=0;x3=0;
    x4=0;x5=0;x6=0;x7=0;
    x8=0;x9=0;
    #s = np.arange(x.shape[0])
    #np.random.shuffle(s)
    #x = x[s];
    #y = y[s];
    for i in range(x.shape[0]):
        
        indx =y[i]

        if(arrx[indx] == 10):
            arrx[indx] = 0
        z = int(arrx[indx])
        if(countx[z] == 6000):
            while(countx[z] == 6000):
                arrx[indx] = arrx[indx] + 1
                if(arrx[indx] == 10):
                    arrx[indx] = 0
                z = int(arrx[indx])
                if(countx[z] < 6000):
                    break;
        


        if(arrx[indx] == 0):
            xblock0[x0] = x[i]
            yblock0[x0] = y[i]
            countx[0] = countx[0] + 1
            arrx[indx] = arrx[indx] + 1
            x0 = x0 + 1

        elif(arrx[indx] ==1):
            xblock1[x1] = x[i]
            yblock1[x1] = y[i]
            countx[1] = countx[1] + 1 
            arrx[indx] = arrx[indx] + 1
            x1 = x1 + 1

        elif(arrx[indx] == 2):
            xblock2[x2] = x[i]
            yblock2[x2] = y[i]
            countx[2] = countx[2] + 1
            arrx[indx] = arrx[indx] + 1
            x2 = x2 + 1

        elif(arrx[indx] == 3):
            xblock3[x3] = x[i]
            yblock3[x3] = y[i]
            countx[3] = countx[3] + 1
            arrx[indx] = arrx[indx] + 1
            x3 = x3 + 1

        elif(arrx[indx] == 4):
            xblock4[x4] = x[i]
            yblock4[x4] = y[i]
            countx[4] = countx[4] + 1
            arrx[indx] = arrx[indx] + 1
            x4 = x4 + 1

        elif(arrx[indx] == 5):
            xblock5[x5] = x[i]
            countx[5] = countx[5] + 1
            arrx[indx] = arrx[indx] + 1
            x5 = x5 + 1

        elif(arrx[indx] == 6):
            xblock6[x6] = x[i]
            yblock6[x6] = y[i]
            countx[6] = countx[6] + 1
            arrx[indx] = arrx[indx] + 1
            x6 = x6 + 1

        elif(arrx[indx] == 7):
            xblock7[x7] = x[i]
            yblock7[x7] = y[i]
            countx[7] = countx[7] + 1
            arrx[indx] = arrx[indx] + 1
            x7 = x7 + 1

        elif(arrx[indx] == 8):
            xblock8[x8] = x[i]
            yblock8[x8] = y[i]
            countx[8] = countx[8] + 1
            arrx[indx] = arrx[indx] + 1
            x8 = x8 + 1

        elif(arrx[indx] == 9):
            xblock9[x9] = x[i]
            yblock9[x9] = y[i]
            countx[9] = countx[9] + 1
            arrx[indx] = arrx[indx] + 1
            x9 = x9 + 1   

    for k in range(K):
        if(k==0):
            val_data = xblock0
            val_label = yblock0

            train_data = np.concatenate((xblock1,xblock2,xblock3,xblock4,xblock5,xblock6,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock1,yblock2,yblock3,yblock4,yblock5,yblock6,yblock7,yblock8,yblock9))
        elif(k==1):
            val_data = xblock1
            val_label = yblock1

            train_data = np.concatenate((xblock0,xblock2,xblock3,xblock4,xblock5,xblock6,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock2,yblock3,yblock4,yblock5,yblock6,yblock7,yblock8,yblock9))
        elif(k==2):
            val_data = xblock2
            val_label = yblock2

            train_data = np.concatenate((xblock0,xblock1,xblock3,xblock4,xblock5,xblock6,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock3,yblock4,yblock5,yblock6,yblock7,yblock8,yblock9))
        elif(k==3):
            val_data = xblock3
            val_label = yblock3

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock4,xblock5,xblock6,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock4,yblock5,yblock6,yblock7,yblock8,yblock9))
        elif(k==4):
            val_data = xblock4
            val_label = yblock4

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock3,xblock5,xblock6,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock3,yblock5,yblock6,yblock7,yblock8,yblock9))
        elif(k==5):
            val_data = xblock5
            val_label = yblock5

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock3,xblock4,xblock6,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock3,yblock4,yblock6,yblock7,yblock8,yblock9))
        elif(k==6):
            val_data = xblock6
            val_label = yblock6

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock3,xblock4,xblock5,xblock7,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock3,yblock4,yblock5,yblock7,yblock8,yblock9))
        elif(k==7):
            val_data = xblock7
            val_label = yblock7

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock3,xblock4,xblock5,xblock6,xblock8,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock3,yblock4,yblock5,yblock6,yblock8,yblock9))
        elif(k==8):
            val_data = xblock8
            val_label = yblock8

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock3,xblock4,xblock5,xblock6,xblock7,xblock9))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock3,yblock4,yblock5,yblock6,yblock7,yblock9))
        elif(k==9):
            val_data = xblock9
            val_label = yblock9

            train_data = np.concatenate((xblock0,xblock1,xblock2,xblock3,xblock4,xblock5,xblock6,xblock7,xblock8))
            train_label = np.concatenate((yblock0,yblock1,yblock2,yblock3,yblock4,yblock5,yblock6,yblock7,yblock8))
        print('Fold number: ',k)
        strain = np.arange(train_data.shape[0])
        np.random.shuffle(strain)
        train_data = x[strain];
        train_label = y[strain];

        stest = np.arange(val_data.shape[0])
        np.random.shuffle(stest)
        val_data = x[stest];

        val_label = y[stest];
        
        for i, params in enumerate(parameters):
            l = Algorithm(params)
            learner.rbfn(new_x_train,new_y_train)
            pred = learner.rbfn_test(new_x_test)
            a = getaccuracy(new_y_test,pred)
            acc = g_accuracy(new_y_test,pred)
            print(acc)
            all_acc[i,k] = a
            all_g[i,k] = acc


    avg_acc = np.mean(all_acc, axis=1)
    avg_g = np.mean(all_g, axis=1)

    print('............avg_acc..........')
    print(avg_acc)
    print('............avg_g..........')
    print(avg_g)


    best_parameters = parameters[0]
    bestacc = avg_g[0]
    for i, params in enumerate(parameters):
        if(avg_g[i] > bestacc):
            bestacc = avg_g[i]
            best_parameters = parameters[i]
    print(best_parameters)
    return best_parameters



if __name__ == '__main__':


    #data
    (x_train, y_train) = loadlocal_mnist(
        images_path ='/home/pathak1/Music/FMNIST/train-images-idx3-ubyte',
        labels_path ='/home/pathak1/Music/FMNIST/train-labels-idx1-ubyte'
        )
    (x_test, y_test) = loadlocal_mnist(
        images_path ='/home/pathak1/Music/FMNIST/t10k-images-idx3-ubyte',
        labels_path ='/home/pathak1/Music/FMNIST/t10k-labels-idx1-ubyte'
        )
    
    
    classalgs = {
        'ConvNNet': algs.convnet,
        }

    ## hyper-parameters
    parameters = {
        'ConvNNet' : [
                {'k':10},
                {'k':25},
                {'k':50},
                {'k':100}
        ],
        
        }

    x_train = x_train.reshape((60000,28,28,1))
    x_train = x_train.astype('float32')/255
    x_test = x_test.reshape((10000,28,28,1))
    x_test = x_test.astype('float32')/255


    best_parameters = {}
    for learnername, Learner in classalgs.items():
        print('running learner = ' + learnername)
        params = parameters.get(learnername)
        #params = best_parameters[learnername]
        learner = Learner(params)
        #Training the model with best hyper parameters
        learner.modelcons()
        learner.learn(x_train, y_train)
        new_x_train,new_y_train = learner.baseconv(60000,x_train,y_train)
        new_x_test,new_y_test  = learner.baseconv(10000,x_test,y_test)
        new_y_test = np.array(new_y_test.tolist())
        new_y_train = np.array(new_y_train.tolist())
        new_y_train = np.reshape(new_y_train,(60000,))
        new_y_test = np.reshape(new_y_test,(10000,))
        print('ytest',new_y_test.shape)
        new_y_train = new_y_train.astype('int')
        new_y_test = new_y_test.astype('int')
        
        best_parameters[learnername] = s_cross_validate(10, new_x_train,new_y_train, Learner, params)  
    


    for learnername, Learner in classalgs.items():
        print('running learner = ' + learnername)
        params = parameters.get(learnername)
        params = best_parameters[learnername]
        learner = Learner(params)
        #Training the model with best hyper parameters
        learner.modelcons()
        learner.learn(x_train, y_train)
        new_x_train,new_y_train = learner.baseconv(60000,x_train,y_train)
        new_x_test,new_y_test  = learner.baseconv(10000,x_test,y_test)
        new_y_test = np.array(new_y_test.tolist())
        new_y_train = np.array(new_y_train.tolist())
        new_y_train = np.reshape(new_y_train,(60000,))
        new_y_test = np.reshape(new_y_test,(10000,))
        print('ytest',new_y_test.shape)
        new_y_train = new_y_train.astype('int')
        new_y_test = new_y_test.astype('int')

        learner.rbfn(new_x_train,new_y_train)
        pred = learner.rbfn_test(new_x_test)
        acc = g_accuracy(new_y_test,pred)
        print(acc)
        # Test model
        best_acc = learner.predict(x_test,y_test)

    print("BEST ACCURACY:",best_acc)

                