from audioop import cross
from scipy import misc
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.image as mpimg
import os;
import time;
import datetime;
from matplotlib import pyplot
# import cv2;
# #compatibility for tf v2.0
# if(tf.__version__.split('.')[0]=='2'):
#     import tensorflow.compat.v1 as tf
#     tf.disable_v2_behavior()    
# --------------------------------------------------
# setup

# plt.figure(figsize=(10,10))
# path=os.getcwd()+'/CIFAR10/Train/0';
# test_folder=path
# IMG_WIDTH=200
# IMG_HEIGHT=200
# img_folder=os.getcwd()+'/CIFAR10/Train/0';
# for i in range(5):
#     file = random.choice(os.listdir(img_folder))
#     image_path= os.path.join(img_folder, file)
#     img=mpimg.imread(image_path)
#     ax=plt.subplot(1,5,i+1)
#     ax.title.set_text(file)
#     # print(img)
#     plt.imshow(img,cmap='gray')
#     plt.show()

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
    return tf.Variable(W);


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    # b=tf.random.truncated_normal(shape,stddev=0.1);
    b=tf.constant(0.1,shape=shape);
    # initial=tf.keras.initializers.GlorotUniform();
    # b=initial(shape);
    return tf.Variable(b);


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


def getData(type_data,size):
    directory=os.getcwd()+'/CIFAR10/'+type_data;
    data=tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        validation_split=0.0,
        color_mode="grayscale",
        batch_size=size,
        seed=None,
        image_size=(28, 28),
        shuffle=True
    )
    scaler=Rescaling(1./255, offset=0.0)
    normalised_data=data.map(lambda x,y: (scaler(x),y));
    return normalised_data;
    # return data

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

def main():
    # Specify training parameters
    max_step = 5000 # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time() # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK
    W_conv1 = weight_variable([5,5,1,32]);
    b_conv1 = weight_variable([32]);
  
    W_conv2 = weight_variable([5,5,32,64]);
    b_conv2 = weight_variable([64]);

    W_fc1 = weight_variable([7 * 7 * 64, 1024]);
    b_fc1 =weight_variable([1024]);

    W_fc2 =weight_variable([1024, 10]);
    b_fc2 =bias_variable([10]);
    
    train_summary_dir = 'log/train/'
    train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
    train_data=getData("Train",32)
    test_data=getData("Test",1000)
    # for x,y in test_data.take(1):
    #     print(x.shape)
    it=iter(train_data);
    test_dir = 'log/test/'
    test_summary_writer = tf.summary.create_file_writer(test_dir)
    train_step = tf.keras.optimizers.Adam(learning_rate=1e-3);
    # train_step = tf.train.AdagradOptimizer(1e-4)
    # train_step = tf.train.MomentumOptimizer(learning_rate=1e-4,momentum=0.9)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
    # train_step=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=False);
    # res=set();
    for i in range(max_step):
        with train_summary_writer.as_default(step=i):

            if(i==max_step-1):
                for l,k in test_data.take(1):
                    image_batch=l;
                    labels_batch = k;
            else:
                try:
                    image_batch, labels_batch = next(it)
                except:
                    it=iter(train_data)
                    image_batch, labels_batch = next(it)

            x=image_batch;
            # print(x.shape)
            y=labels_batch; 
            # print(y)
            with tf.GradientTape() as tape:
                Z_conv1=conv2d(x, W_conv1) + b_conv1;
                h_conv1 = tf.nn.relu(Z_conv1);
                # h_conv1 = tf.nn.tanh(Z_conv1);
                # h_conv1 = tf.nn.sigmoid(Z_conv1);
                # h_conv1 = tf.nn.leaky_relu(Z_conv1);
                h_pool1 = max_pool_2x2(h_conv1)

                # Convolutional layer 2
                Z_conv2=conv2d(h_pool1, W_conv2) + b_conv2;
                h_conv2 = tf.nn.relu(Z_conv2)
                # h_conv2 = tf.nn.tanh(Z_conv2)
                # h_conv2 = tf.nn.sigmoid(Z_conv2)
                # h_conv2 = tf.nn.leaky_relu(Z_conv2)
                h_pool2 = max_pool_2x2(h_conv2)

                # Fully connected layer 1
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

                Z_fc1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1;
                h_fc1 = tf.nn.relu(Z_fc1);
                # h_fc1 = tf.nn.tanh(Z_fc1);
                # h_fc1 = tf.nn.sigmoid(Z_fc1);
                # h_fc1 = tf.nn.leaky_relu(Z_fc1);
                # Dropout
                # keep_prob  = tf.placeholder(tf.float32)
                # h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

                # Fully connected layer 2 (Output layer)
                Z_fc2=tf.matmul(h_fc1, W_fc2) + b_fc2;
                y_conv = tf.nn.softmax(Z_fc2, name='y_conv')
                # cross_entropy =tf.reduce_mean(-tf.reduce_sum(y* tf.math.log(y_conv)));
                cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z_fc2,labels=y))
                #different cross entropy- logits
                #+0.002*(tf.math.reduce_sum(tf.square(W_conv1))+tf.math.reduce_sum(tf.square(W_conv2))+tf.math.reduce_sum(tf.square(W_fc1))+tf.math.reduce_sum(tf.square(W_fc2)));

            if(i%500==0):
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
                tf.summary.scalar('loss', cross_entropy);
   
    

            # if(i%1100 !=0):
            grads=tape.gradient(cross_entropy,[W_conv1,W_conv2,b_conv1,b_conv2,W_fc1,W_fc2,b_fc1,b_fc2]);
                # print(grads)
                # gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]

            train_step.apply_gradients(zip(grads,[W_conv1,W_conv2,b_conv1,b_conv2,W_fc1,W_fc2,b_fc1,b_fc2]));
            # print(train_step.weights)
            correct_prediction =tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1));
            accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy');
            

            if i%500 == 0 :
                print("step %g train accuracy %g"%(i,accuracy));
                # print(y_conv);
                train_summary_writer.flush()
            # save the checkpoints every 1100 iterations, i % 1100 == 0 or 
            if i == max_step-1:
                with test_summary_writer.as_default(step=i):
                    tf.summary.scalar('accuracy', accuracy)
                print("step %g test accuracy %g"%(i,accuracy));

    stop_time = time.time()
    print('The training takes %f second to finish'%(stop_time - start_time))

    n_filters, ix = 32, 1
    pyplot.figure(figsize=(10,10))
    for i in range(n_filters):
        # get the filter
        f = W_conv1[:, :, :, i]
        # plot each channel separately
        for j in range(1):
            # specify subplot and turn of axis
            ax = pyplot.subplot(7, 5, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
        # show the figure
    pyplot.show()
    # print(len(res))
if __name__ == "__main__":
    main()



