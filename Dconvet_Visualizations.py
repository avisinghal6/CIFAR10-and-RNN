from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import os;
import tensorflow_addons as tfa
# --------------------------------------------------
# setup
if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    

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


ntrain =  500# per class
ntest =  100# per class
nclass =  10# number of classes
imsize = 28
nchannels = 1
batchsize = 5

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
# W_conv1 = weight_variable([5, 5, 1, 32])
W_conv1=tf.get_variable(name="W_conv1",shape=[5, 5, 1, 32])
b_conv1=tf.get_variable(name="B_conv1",shape=[32])
# b_conv1 = bias_variable([32])
    
Z_conv1=conv2d(tf_data, W_conv1) + b_conv1;
h_conv1 = tf.nn.relu(Z_conv1); 
# h_conv1 = tf.nn.tanh(Z_conv1);
# h_conv1 = tf.nn.sigmoid(Z_conv1);
# h_conv1 = tf.nn.leaky_relu(Z_conv1);
h_pool1, max_index1 = tf.nn.max_pool_with_argmax(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
# unpooling1 = tfa.layers.MaxUnpooling2D()(h_pool1, max_index1)


# Convolutional layer 2
W_conv2=tf.get_variable(name="W_conv2",shape=[5, 5, 32, 64])
b_conv2=tf.get_variable(name="B_conv2",shape=[64])

Z_conv2=conv2d(h_pool1, W_conv2) + b_conv2;
h_conv2 = tf.nn.relu(Z_conv2)

h_pool2, max_index2 = tf.nn.max_pool_with_argmax(h_conv2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

unpooling2 = tfa.layers.MaxUnpooling2D()(h_pool2, max_index2)
unrelu2= tf.nn.relu(unpooling2);
dconv2=conv2d(unrelu2, tf.transpose(W_conv2,perm=[1,0,3,2]));
# dconv2=conv2d(unrelu2-b_conv2, tf.transpose(W_conv2,perm=[1,0,3,2]));

unpooling1 = tfa.layers.MaxUnpooling2D()(dconv2, max_index1)
unrelu1= tf.nn.relu(unpooling1);
dconv1=conv2d(unrelu1, tf.transpose(W_conv1,perm=[1,0,3,2]));

Unpooling1 = tfa.layers.MaxUnpooling2D()(h_pool1, max_index1)
Unrelu1= tf.nn.relu(Unpooling1);
Dconv1=conv2d(Unrelu1, tf.transpose(W_conv1,perm=[1,0,3,2]));
# Dconv1=conv2d(Unrelu1-b_conv1, tf.transpose(W_conv1,perm=[1,0,3,2]));

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1=tf.get_variable(name="W_fc1",shape=[7 * 7 * 64, 1024])
b_fc1=tf.get_variable(name="B_fc1",shape=[1024])

Z_fc1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1;
h_fc1 = tf.nn.relu(Z_fc1);

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2=tf.get_variable(name="W_fc2",shape=[1024,10])
b_fc2=tf.get_variable(name="B_fc2",shape=[10])
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

Z_fc2=tf.matmul(h_fc1_drop, W_fc2) + b_fc2;
y = tf.nn.softmax(Z_fc2, name='y')

# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
save_path = saver.restore(sess, "SAVE/weights.ckpt")
batch_xs = np.zeros((batchsize,28,28,1))#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize,10))#setup as [batchsize, the how many classes] 
for i in range(1): # try a small iteration size once it works then continue
    perm = np.arange(1000)
    np.random.shuffle(perm)
    # print(perm[0])
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Test[perm[j],:,:,:]
        batch_ys[j,:] = LTest[perm[j],:]

    acc=accuracy.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 1}) # dropout only during training
    print("accuracy",acc);

# --------------------------------------------------
ix=1
plt.figure(figsize=(10,10))
for i in range(batchsize):
        # get the filter
    
    t1=Dconv1.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 1})
    t2=dconv1.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 1})
    # print(t[:,:,:,1])
    # print(t2[:,:,:,1])
    f1 = t1[i, :, :, 0]
    f2= t2[i, :, :, 0]
    # print(f.shape)
        # plot each channel separately
    for j in range(1):
            # specify subplot and turn of axis
        
        ax = plt.subplot(5,3, ix)
        # plot filter channel in grayscale
        plt.title("Dconv1")
        plt.imshow(f1[:,:], cmap='gray')
        ix += 1
        ax = plt.subplot(5,3, ix)
        plt.title("Original Image 1")
        plt.imshow(batch_xs[i,:,:,0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ix += 1
        ax = plt.subplot(5,3, ix)
        plt.title("Dconv2")
        plt.imshow(f2[:,:], cmap='gray')
        ix += 1
        
        # show the figure
plt.show()

sess.close()

