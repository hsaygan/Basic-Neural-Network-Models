#XOR Function using Machine Learning
import tensorflow as tf
import numpy as np

X = ([[0,0],
    [0,1],
    [1,0],
    [1,1]])

Y = ([[1,0],        #One-Hot-Array  ie. 0 = [1,0] and 1 = [0,1]
    [0,1],
    [0,1],
    [1,0]])

input_nodes = 2
hl1_nodes = 4
hl2_nodes = 4
no_class = 2
n_epochs = 9000

x = tf.placeholder('float', [None, 2])
y = tf.placeholder('float', [None, 2])

def neural_network_model(data):
    hl1 = {'weights':tf.Variable(tf.random_normal([2, hl1_nodes])),
                      'biases':tf.Variable(tf.random_normal([hl1_nodes]))}
    hl2 = {'weights':tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                      'biases':tf.Variable(tf.random_normal([hl2_nodes]))}
    ol = {'weights':tf.Variable(tf.random_normal([hl2_nodes, no_class])),
                      'biases':tf.Variable(tf.random_normal([no_class]))}

    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, ol['weights']) + ol['biases']
    return output

def train_and_test(data):
    prediction = neural_network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as sess:
        #Training Data
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
            epoch_loss += c
            if (epoch%100==0):
                print("Epoch", epoch+1, "completed out of", n_epochs, " --- Loss :", epoch_loss)

        #Testing Data
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:X, y:Y}))

train_and_test(x)
