from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import os;
# --------------------------------------------------
# setup
if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    W=tf.random.truncated_normal(shape,stddev=0.1);
    # initial=tf.keras.initializers.GlorotUniform();
    # W=initial(shape);
    return tf.Variable(W,name="W");
    

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE

    b=tf.constant(0.1,shape=shape);
    return tf.Variable(b,name="B");

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE

    h_conv=tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME");
    return h_conv;

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE

    h_max=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME");
    
    return h_max


def max_summary(x,name):
    tf.summary.scalar(name+" MAX", tf.reduce_max(x));
def min_summary(x,name):
    tf.summary.scalar(name+" MIN", tf.reduce_min(x));
def mean_summary(x,name):
    tf.summary.scalar(name+" MEAN", tf.reduce_mean(x));
def std_summary(x,name):
    tf.summary.scalar(name+ "STD", tf.math.reduce_std(x));
def hist_summary(x,name):
    tf.summary.histogram(name + "HISTOGRAM", x);

def cr_summary(x,name):
    max_summary(x,name);
    min_summary(x,name);
    mean_summary(x,name);
    std_summary(x,name);
    hist_summary(x,name);


ntrain =  500# per class
ntest =  100# per class
nclass =  10# number of classes
imsize = 28
nchannels = 1
batchsize = 50

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
directory=os.getcwd()+'/CIFAR10/';
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = directory+'Train/%d/Image%05d.png' % (iclass,isample)
        im = plt.imread(path); # 28 by 28
        # im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = directory+'Test/%d/Image%05d.png' % (iclass,isample)
        im = plt.imread(path); # 28 by 28
        # im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, [None, 28,28,1], name='x')#tf variable for the data, remember shape is [None, width, height, numberOfChannels] 
tf_labels = tf.placeholder(tf.float32, [None, 10],  name='y_') #tf variable for labels
result_dir1 = './results/train' # directory where the results from the training are saved
result_dir2 = './results/test' # directory where the results from the training are saved
# --------------------------------------------------
# model
#create your model
W_conv1=tf.Variable(tf.random.truncated_normal([5, 5, 1, 32],stddev=0.1),name="W_conv1")
# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]),name="B_conv1")
    
Z_conv1=conv2d(tf_data, W_conv1) + b_conv1;
h_conv1 = tf.nn.relu(Z_conv1); 
# h_conv1 = tf.nn.tanh(Z_conv1);
# h_conv1 = tf.nn.sigmoid(Z_conv1);
# h_conv1 = tf.nn.leaky_relu(Z_conv1);
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
# W_conv2 = weight_variable([5, 5, 32, 64])
W_conv2=tf.Variable(tf.random.truncated_normal([5, 5, 32, 64],stddev=0.1),name="W_conv2")
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]),name="B_conv2")
# b_conv2 = bias_variable([64])
Z_conv2=conv2d(h_pool1, W_conv2) + b_conv2;
h_conv2 = tf.nn.relu(Z_conv2)
# h_conv2 = tf.nn.tanh(Z_conv2)
# h_conv2 = tf.nn.sigmoid(Z_conv2)
# h_conv2 = tf.nn.leaky_relu(Z_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1=tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024],stddev=0.1),name="W_fc1")
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]),name="B_fc1")
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

Z_fc1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1;
h_fc1 = tf.nn.relu(Z_fc1);
# h_fc1 = tf.nn.tanh(Z_fc1);
# h_fc1 = tf.nn.sigmoid(Z_fc1);
# h_fc1 = tf.nn.leaky_relu(Z_fc1);
# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2=tf.Variable(tf.random.truncated_normal([1024, 10],stddev=0.1),name="W_fc2")
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]),name="B_fc2")
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

Z_fc2=tf.matmul(h_fc1_drop, W_fc2) + b_fc2;
y = tf.nn.softmax(Z_fc2, name='y')

# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y), reduction_indices=[1]))+0.001*(tf.math.reduce_sum(tf.square(W_conv1))+tf.math.reduce_sum(tf.square(W_conv2))+tf.math.reduce_sum(tf.square(W_fc1))+tf.math.reduce_sum(tf.square(W_fc2)));
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z_fc2,labels=tf_labels));
    
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
c= lambda:tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y), reduction_indices=[1]))
# Training algorithm
# --------------------------------------------------
# optimization
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
# train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.MomentumOptimizer(learning_rate=1e-3,momentum=0.9).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cross_entropy)
# train_step=tf.keras.optimizers.SGD(learning_rate=1e-4).minimize(c,var_list=[W_conv1,W_conv2,W_fc1,W_fc2,b_conv1,b_conv2,b_fc1,b_fc2])
# Add a scalar summary for the snapshot loss.
cost=tf.summary.scalar("Loss", cross_entropy) 
cr_summary(W_conv1,"W_Conv1");
cr_summary(b_conv1,"b_Conv1");
cr_summary(Z_conv1,"Z_Conv1");
cr_summary(h_conv1,"h_Conv1");
cr_summary(h_pool1,"h_pool1");
cr_summary(W_conv2,"W_Conv2");
cr_summary(b_conv2,"b_Conv2");
cr_summary(Z_conv2,"Z_Conv2");
cr_summary(h_conv2,"h_Conv2");
cr_summary(h_pool2,"h_pool2");
cr_summary(W_fc1,"W_fc1");
cr_summary(b_fc1,"b_fc1");
cr_summary(Z_fc1,"Z_fc1");
cr_summary(h_fc1,"h_fc1");
cr_summary(W_fc2,"W_fc2");
cr_summary(b_fc2,"b_fc2");
cr_summary(Z_fc2,"Z_fc2");


summary_op = tf.summary.merge_all()
summary_op2 = tf.summary.merge([cost])
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(result_dir1, sess.graph)
summary_writer2 = tf.summary.FileWriter(result_dir2, sess.graph)
saver=tf.train.Saver()
batch_xs = np.zeros((batchsize,28,28,1))#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize,10))#setup as [batchsize, the how many classes] 
for i in range(2000): # try a small iteration size once it works then continue
    perm = np.arange(5000)
    np.random.shuffle(perm)
    # print(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%100 == 0:
        #calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 0.5})
        print("step %d, training accuracy %g"%(i, train_accuracy))
            # Update the events file which is used to monitor the training (in thi case,
            # only the training loss is monitored)
        summary_str = sess.run(summary_op, feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

    train_step.run(feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 0.5}) # dropout only during training

save_path = saver.save(sess, "SAVE/weights.ckpt")
# --------------------------------------------------
# test




print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
n_filters, ix = 32, 1
plt.figure(figsize=(10,10))
for i in range(n_filters):
        # get the filter
    t=W_conv1.eval();
    f = t[:, :, :, i]
        # plot each channel separately
    for j in range(1):
            # specify subplot and turn of axis
        ax = plt.subplot(7, 5, ix)
        ax.set_xticks([])
        ax.set_yticks([])
            # plot filter channel in grayscale
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
        # show the figure
plt.show()

sess.close()