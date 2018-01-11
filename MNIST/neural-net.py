''' Code Overview :
**  input -> weight & biases -> hidden layer 1 (ACTIVATION FUNCTION)         #Feed Forward
    -> weight & biases -> hl2 (ACTIVATION FUNCTION) -> weight & biases .... -> output layer

**  Compare output to intended output -> COST FUNCTION (cross entropy)

**  Optimization function (OPTIMIZER) -> minimize cost
    (AdamOptimizer... SGD, AdaGrad etc)

**  backpropagation                     #backward motion & manipulation of weights

**  feed forward + backpropagation = epoch      #epoch is a cycle (trying to lower cost)
'''

import tensorflow as tf         #Tensors = MULTI-DIMENTIONAL ARRAYS
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

'''
    So this Example Database has 2 Subparts :
    1. Training Folder
        1.1 Images
        1.2 Labels
    2. Testing Folder
        2.1 Images
        2.2 Labels
    Images are of the format :  ----------------------------    ie. 28 x 28 (784) Grid of Floats, in greyscale [DATA]
                               | 0 0.7 0.2 0.3 1 0 0 0 0
                               | 0 0 0 0  ....
                               | ....
                               | ...
    Lables are of the format :  [1,0,0,0,0,0,0,0,0,0] --denotes--> 0  ie. one_hot format. [CLASSES]
                                [0,1,0,0,0,0,0,0,0,0] --denotes--> 1
                                [0,0,1,0,0,0,0,0,0,0] --denotes--> 2
                                [0,0,0,1,0,0,0,0,0,0] --denotes--> 3
                                        . . .
    Every image corresponds to it's label.
    Training Data trains the neural net to recognise and is used for prediction on testing data.
'''


n_nodes_hl1 = 500        #3 hidden layers
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100        #100 images (example) at a time

#Placeholder for data to be shoved
x = tf.placeholder('float', [None, 784]) #data
        #NONE : 50,000 samples of data, we just didn't specify
        #Creating a 28 * 28 grid = 784 pixels of the image into a 1D ARRAY
y = tf.placeholder('float') #Label of the data

def neural_network_model(data):
    #Input Data * weight + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                        'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                        'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                        'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases' : tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)                                                                     #Activation Function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output
''' Creating kind of data structure to store WEIGHTS and BIASES for each EDGE
        between 2 layers.
        x1
        x2         hl1 n1
        x3  --->   hl1 n2 --->
        .          hl1 n3
        .             .
        .             .
        x784       hl1 n500
        Edge : the connection between each input x and node n
'''

'''     Nodes in output layer is basically the number of Classes/Clusters/Defination we defined as output
        in this case,
        0,1,2,3,4,5,6,7,8,9
        ie. what possible image concedes
'''

''' Returns classes like :
        1,0,0,0,0,0,0,0,0,0 or
        0,1,0,0,0,0,0,0,0,0 or
            .  .  .
        0,0,0,0,0,0,0,0,0,1
'''

def train_neural_network(x):        #what you wanna do with the model
    print("***** Inside the train_neural_network(data) function")
    prediction = neural_network_model(x)
    #cost = tf.reduce_mean(tf.nm.softmax_cross_entropy_with_logits(prediction, y))          #OLD
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
            #Determines loss

    #learning rate (parameter in optimizer Adam = 0.001 (default))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10      #how many epochs : (cycles) feed-forward + backpropagation
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())                #OLD
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):        #_ = variable we dont give a shit about
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)                       #Have to build yourself
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                    #Modifies weights and biases* from neural_network_model automatically'''
                    #TOO HIGH LEVEL!'''
                epoch_loss += c
            print("Epoch : ", epoch, " completed out of ", hm_epochs, "  loss : ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                    #argmax returns index values of '1' in prediction and y (label) and compares.
                    #Returns an array, contains each time it predicted correctly, or if didn't
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    #Cast changes a variable to a particular type
                    #Takes mean of the correct
        print("Accuracy : ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})*100, "%")
                    #Automatically gets testing data to run through our model and get the accuracy'''
                                #TOO HIGH LEVEL!

train_neural_network(x)
