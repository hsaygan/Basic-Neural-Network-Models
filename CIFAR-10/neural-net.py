import tensorflow as tf
import numpy as np
import pickle

#5 Training Batches - 1 Testing Batch
file_name = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

#10 Classes
label_name = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

training_data = []
training_labels = []
testing_data = []
testing_labels = []

input_nodes = 3072  # 1024x1024x1024 RGB
n_hl1_nodes = 4000
n_hl2_nodes = 2000
n_class = 10        # One-Hot-Array of 10 classification
n_epochs = 10
batch_size = 100 #100 Images per batch

with open("dataset/%s"%(file_name[0]), 'rb') as file:
    source = pickle.load(file, encoding='bytes')    #10,000 Images

    # training_data = tf.reshape(source[b'data'], shape=[10000, 3, 1024])
    training_data = source[b'data']
    training_data = training_data.reshape(10000, 3072)
    print ("#Training Data is stored to a the Global Variable")

    # training_labels = tf.one_hot(source[b'labels'], 10)      #Creating One-Hot-Array
    # training_labels = tf.reshape(training_labels, shape=[10000, 10])
    training_labels = np.array([source[b'labels']]).reshape(-1)
    training_labels = np.eye(n_class)[training_labels]
    print ("#Training Labels are stored to a the Global Variable")


with open("dataset/%s"%(file_name[5]), 'rb') as file:
    source = pickle.load(file, encoding='bytes')

    testing_data = source[b'data']
    testing_data.reshape(10000, 3072)
    print ("#Testing Data is stored to a the Global Variable")

    testing_labels = np.array([source[b'labels']]).reshape(-1)
    testing_labels = np.eye(n_class)[testing_labels]
    print ("#Testing Labels are stored to a the Global Variable")

print ("Data : ", training_data)
print ("Labels : ", training_labels)

x = tf.placeholder('float', [None, input_nodes])
y = tf.placeholder('float', [None, n_class])

def neural_network_model(x):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([input_nodes, n_hl1_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_hl1_nodes]))}
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_hl1_nodes, n_hl2_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_hl2_nodes]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_hl2_nodes, n_class])),
                      'biases':tf.Variable(tf.random_normal([n_class]))}

    hl1 = tf.add(tf.matmul(x, hidden_layer_1['weights']), hidden_layer_1['biases'])
    hl1 = tf.nn.relu(hl1)

    hl2 = tf.add(tf.matmul(hl1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    hl2 = tf.nn.relu(hl2)

    output = tf.matmul(hl2, output_layer['weights']) + output_layer['biases']
    return output

def train_and_test(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        #Training Data
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            range_thingy = 10000//batch_size
            for i in range(range_thingy):
                start = i*batch_size
                end = (i+1)*batch_size
                data_batch = np.array(training_data[start:end])
                labels_batch = np.array(training_labels[start:end])
                if (start%1000==0):
                    print ("Data Batch : \n%d to %d" % (start, end))
                _, c = sess.run([optimizer, cost], feed_dict = {x: data_batch, y: labels_batch})
                epoch_loss += c
            print("Epoch", epoch+1, "completed out of", n_epochs, " --- Loss :", epoch_loss)

        #Testing Data
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:testing_data, y:testing_labels}))

train_and_test(x)
