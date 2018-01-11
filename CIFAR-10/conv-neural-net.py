import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from extract_data import get_data, get_info

training_data, training_labels, testing_data, testing_labels = get_data(1)
image_width, image_height, image_channels, n_training_images = get_info(1)

print ("\n\n==================================================================\n\n")

n_class = 10                            #Number of Classification
n_epochs = 10                           #Number of Iterations
batch_size = 1000                       #Images per Batch
keep_rate = 0.8                         #Keep Rate for Neural Network Dropout
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder('float', [None, image_height, image_width, image_channels])
Y = tf.placeholder('float', [None, n_class])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
                        #Image Number, X move, Y move, Channels
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,1,1,1], strides=[1,2,2,1], padding='SAME')
                         #size of window   movement of window

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc1':tf.Variable(tf.random_normal([8*8*64,1024])),
               'W_fc2':tf.Variable(tf.random_normal([1024,128])),
               'out':tf.Variable(tf.random_normal([128, n_class]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc1':tf.Variable(tf.random_normal([1024])),
               'b_fc2':tf.Variable(tf.random_normal([128])),
               'out':tf.Variable(tf.random_normal([n_class]))}

    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    conv1 = tf.nn.relu(tf.add(conv2d(x, weights['W_conv1']), biases['b_conv1']))
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(tf.add(conv2d(conv1, weights['W_conv2']), biases['b_conv2']))
    conv2 = maxpool2d(conv2)

    fc1 = tf.reshape(conv2,[-1, 8*8*64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['W_fc1']), biases['b_fc1']))

    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['W_fc2']), biases['b_fc2']))
    #fc2 = tf.nn.dropout(fc2, keep_rate)

    output = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return output

def train_and_test(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #Training Model (Epoch by Epoch)
        for epoch in range(n_epochs):
            batchy_batch = n_training_images//batch_size
            epoch_loss = 0
            for i in range(batchy_batch):
                start = i*batch_size
                end = (i+1)*batch_size
                data_batch = np.array(training_data[start:end])
                labels_batch = np.array(training_labels[start:end])

                #Verbose
                if (start%1000==0):
                    print ("\n\n======= DATA BATCH ===== %d to %d OUT OF %d" % (start, end, n_training_images))
                    # print ("\nData [0] : ", data_batch[0])
                    # print ("\nLabels [0] : ", labels_batch[0])

                    # print("\nExample of feeded data : (In The PyPlot)")
                    # display_data = data_batch.astype("uint8")
                    # fig, axes1 = plt.subplots(2,5,figsize=(3,3))
                    # i = 0
                    # for j in rangde(2):
                    #     for k in range(5):
                    #         #i = np.random.choice(range(len(display_data)))
                    #         axes1[j][k].set_axis_off()
                    #         axes1[j][k].imshow(display_data[i:i+1][0])
                    #         temp_label = list(labels_batch[i])
                    #         axes1[j][k].set_title(label_name[temp_label.index(1)])
                    #         i = i+1
                    #
                    # plt.show(block=False)
                    # plt.pause(2)
                    # plt.close()

                _, c = sess.run([optimizer, cost], feed_dict = {X:data_batch, Y:labels_batch})
                epoch_loss += c
            print("\n:::::::::::::: EPOCH", epoch+1, "completed out of", n_epochs, " --- Loss :", epoch_loss, "::::::::::::::")

        # #Training Model (Epoch by Epoch)
        # for batch in range(5):
        #     batch_loss = 0
        #     for epoch in range(n_epochs):
        #         start = batch*batch_size
        #         end = (batch+1)*batch_size
        #         data_batch = np.array(training_data[start:end])
        #         labels_batch = np.array(training_labels[start:end])
        #
        #         #Verbose
        #         if (start%1000==0):
        #             print ("\n\n======= DATA BATCH ===== %d to %d OUT OF %d" % (start, end, n_training_images))
        #             # print ("\nData [0] : ", data_batch[0])
        #             # print ("\nLabels [0] : ", labels_batch[0])
        #
        #             # print("\nExample of feeded data : (In The PyPlot)")
        #             # display_data = data_batch.astype("uint8")
        #             # fig, axes1 = plt.subplots(2,5,figsize=(3,3))
        #             # i = 0
        #             # for j in range(2):
        #             #     for k in range(5):
        #             #         #i = np.random.choice(range(len(display_data)))
        #             #         axes1[j][k].set_axis_off()
        #             #         axes1[j][k].imshow(display_data[i:i+1][0])
        #             #         temp_label = list(labels_batch[i])
        #             #         axes1[j][k].set_title(label_name[temp_label.index(1)])
        #             #         i = i+1
        #             #
        #             # plt.show(block=False)
        #             # plt.pause(2)
        #             # plt.close()
        #
        #         _, c = sess.run([optimizer, cost], feed_dict = {X: data_batch, Y: labels_batch})
        #         batch_loss += c
        #     print("\n:::::::::::::: Batch", batch+1, "completed out of", 5, " --- Loss by batch :", batch_loss, "::::::::::::::")

        # #Saving Model
        # savepath = saver.save(sess, 'models/cifar-10-latest-model-session.ckpt')
        # print("\n\nSaved the Model as '", savepath, "'.\n\n")

        # #Load Model
        # saver.restore(sess, 'models/cifar-10-latest-model-session.ckpt')
        # print("\n\nLoaded Pre-Saved Model\n\n")

        #Testing Model
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('\nAccuracy:', (accuracy.eval({X:testing_data, Y:testing_labels}) * 100), "%")

train_and_test(X)
