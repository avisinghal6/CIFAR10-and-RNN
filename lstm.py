from re import I
import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 

if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior() 

# print(tf.__version__)
# import tensorflow_datasets

(X_train, y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
# X_train = x_train.reshape(x_train.shape[0], 28, 28,1)
X_train, X_test=X_train.astype('float32'),X_test.astype('float32')
X_train, X_test = X_train/255.0, X_test/255.0
y_train = tf.keras.utils.to_categorical(y_train,10)
# X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
Y_test = tf.keras.utils.to_categorical(Y_test,10)
# print(type(X_train))

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_dataset = train_dataset.batch(50)
# # print(train_dataset)
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
# # test_dataset = test_dataset.batch(50)

result_dir1 = './Results_LSTM/Train'
result_dir2 = './Results_LSTM/Test'
learningRate = 1e-3
trainingIters = 5500
batchSize = 50
displayStep = 100

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 5#number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

# print(X_test.shape)
testData = X_test.reshape((-1, nSteps, nInput))
testLabel = Y_test

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels
    # lstmCell = tf.nn.rnn_cell.LSTMCell(num_units=nHidden,activation='relu')
    # lstmCell = tf.nn.rnn_cell.BasicRNNCell(num_units=nHidden,activation='relu')
    lstmCell = tf.nn.rnn_cell.GRUCell(num_units=nHidden,activation='relu')
    outputs, state = tf.nn.static_rnn(lstmCell, x,dtype=tf.float32)
    # lstmCell = rnn_cell.BasicRNNCell(nHidden)#find which lstm to use in the documentation
    # outputs, states = lstmCell.apply(x) #for the rnn where to get the output and hidden state 

    
    return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name='accuracy')

tf.summary.scalar("Loss", cost)
tf.summary.scalar("Accuracy", accuracy)

summary_op = tf.summary.merge_all()
summary_op2 = tf.summary.merge_all()
init = tf.initialize_all_variables()
# it= train_dataset.make_one_shot_iterator()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(result_dir1, sess.graph)
    summary_writer2 = tf.summary.FileWriter(result_dir2, sess.graph)
    sess.run(init)
    step = 0
    iter=0
    while iter<trainingIters :
        if step==1200:
            step=0;
        # print(step)
        batchX= X_train[step*batchSize:(step+1)*batchSize,:]  #mnist has a way to get the next batch
        # print(batchX.shape)
        batchX = batchX.reshape((batchSize, nSteps, nInput))
        batchY=y_train[step*batchSize:(step+1)*batchSize,:]
        sess.run(optimizer, feed_dict={x:batchX, y:batchY})
        # print(weights['out']);
        if iter % displayStep == 0:
            acc = accuracy.eval(feed_dict={x:batchX, y:batchY})
            loss = cost.eval(feed_dict={x:batchX, y:batchY})
            print("Iter " + str(iter) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            summary_str = sess.run(summary_op, feed_dict={x:batchX, y:batchY})
            summary_writer.add_summary(summary_str, iter)
            summary_writer.flush()
        step +=1
        iter+=1;
        
        if iter%1100 ==0:
            print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x:testData,y:testLabel}))
            summary_str = sess.run(summary_op, feed_dict={x:testData,y:testLabel})
            summary_writer2.add_summary(summary_str, iter)
            summary_writer2.flush()
        # print(step)
    # print('Optimization finished')

    
    