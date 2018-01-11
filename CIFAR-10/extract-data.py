import pickle
import numpy as np

#5 Training Batches - 1 Testing Batch
file_name = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

#10 Classifications
label_name = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

images_per_file = 10000     #Number of Images per File
image_width = 32            #X Axis
image_height = 32           #Y Axis
image_channels = 3          #RGB
n_class = 10                #One-Hot-Array of 10 classification

#Importing Training Data
def get_training_data(count):
    training_data = np.zeros(shape=(count*images_per_file, image_height, image_width, image_channels))
    training_labels = np.zeros(shape=(count*images_per_file, 10))
    for current_file in file_name[:count]:
        print ("\n\nWorking with", current_file, "as Training Data")
        with open("dataset/%s"%(current_file), 'rb') as file:
            source = pickle.load(file, encoding='bytes')
            if (current_file==file_name[0]):
                temp_data = np.array(source[b'data']).reshape(-1, image_channels, image_width, image_height).transpose(0,2,3,1)
                training_data = temp_data.reshape(-1, image_height, image_width, image_channels)
                print ("\n#Training Data is stored to a the Global Variable")

                temp_labels = np.array(source[b'labels'])
                temp_labels = np.eye(n_class)[temp_labels]
                training_labels = temp_labels
                print ("#Training Labels are stored to a the Global Variable")
            else:
                temp_data = np.array(source[b'data']).reshape(-1, image_channels, image_width, image_height).transpose(0,2,3,1)
                training_data = np.concatenate((training_data, temp_data))
                training_data = training_data.reshape(-1, image_height, image_width, image_channels)
                print ("\n#Training Data is stored to a the Global Variable")

                temp_labels = np.array(source[b'labels'])
                temp_labels = np.eye(n_class)[temp_labels]
                training_labels = np.concatenate((training_labels,temp_labels))
                print ("#Training Labels are stored to a the Global Variable")
    return training_data, training_labels

#Importing Testing Data
def get_testing_data():
    with open("dataset/%s"%(file_name[5]), 'rb') as file:
        print ("\n\nWorking with", file_name[5], "as Testing Data")
        source = pickle.load(file, encoding='bytes')

        temp_data = np.array(source[b'data']).reshape(-1, image_channels, image_width, image_height).transpose(0,2,3,1)
        testing_data = temp_data.reshape(-1, image_height, image_width, image_channels)
        print ("\n#Testing Data is stored to a the Global Variable")

        temp_labels = np.array(source[b'labels'])
        temp_labels = np.eye(n_class)[temp_labels]
        testing_labels = temp_labels
        print ("#Testing Labels are stored to a the Global Variable")
    return testing_data, testing_labels

#Export Data Info
def get_info(count):
    return image_width, image_height, image_channels, count*images_per_file

#Export Data
def get_data(count):
    training_data, training_labels = get_training_data(count)
    testing_data, testing_labels = get_testing_data()

    print ("\n\n\TRAINING IMAGES : ")
    print ("\n# Files Used - 1")
    print ("\n# Number of Images - ", len(training_data)) #len(training_labels))
    print ("\n# EXAMPLE (1 Sample Image) : ")
    print ("\nData - \n",  training_data[0])
    print ("\nLabels - \n",training_labels[0])

    print ("\n\n\nTESTING IMAGES : ")
    print ("\n# Files Used - 1")
    print ("\n# Number of Images - ", len(testing_data)) #len(testing_labels))
    print ("\n# EXAMPLE (1 Sample Image) : ")
    print ("\nData - \n",  testing_data[0])
    print ("\nLabels - \n",testing_labels[0])

    return training_data, training_labels, testing_data, testing_labels
