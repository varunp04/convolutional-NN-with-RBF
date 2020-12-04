import numpy as np
# import tensorflow as tf


# x = [0., -1., 2., 3.]
# pred = tf.nn.softmax(x,axis=1)
# arr = np.argmax(pred,axis =1)
# arr = np.ravel(arr)
# print(arr)


test = np.array([1,2,4,3,4,4,4,2,5,1,0])
pred = np.array([1,1,4,3,4,3,4,2,5,1,4])

TP =0

for i in range(len(test)):
	if(test[i] == pred[i]):
		TP = TP + 1

print(TP)



acc = np.divide(TP,10000)
print(acc)


# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([[1,2,3],[6,5,4],[9,9,9]])
# y = np.arange(10)
# for i in range(x.shape[0]):
# 	param = 'HP' + str(i+1) 
# 	plt.plot(x[i],label = param)
# plt.xticks(range(0,9))

# plt.title('Accuracy plot of different parameters vs each fold(0-9)')
# plt.xlabel('Folds')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# from mlxtend.data import loadlocal_mnist
# #data
#     (x_train, y_train) = loadlocal_mnist(
#         images_path ='E:/uofa/SEM2/ECE626/P#1/FMNIST/train-images-idx3-ubyte',
#         labels_path ='E:/uofa/SEM2/ECE626/P#1/FMNIST/train-labels-idx1-ubyte'
#         )
#     (x_test, y_test) = loadlocal_mnist(
#         images_path ='E:/uofa/SEM2/ECE626/P#1/FMNIST/t10k-images-idx3-ubyte',
#         labels_path ='E:/uofa/SEM2/ECE626/P#1/FMNIST/t10k-labels-idx1-ubyte'
#         )


# # x = np.zeros((6000,28,28,1))
# # y = np.ones((2,2,2,2))
# # w = np.array([[0,0],[1,1]])



# # a = np.array([[0,0,0],[1,1,1],[2,2,2]])

# # b = np.array([[0,0],[1,1],[2,2]])

# # s = np.arange(a.shape[0])
# # np.random.shuffle(s)
# # a = a[s]
# # b = b[s]

# # print(x.shape[0])
# # print(b)