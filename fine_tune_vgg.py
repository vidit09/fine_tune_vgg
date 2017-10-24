from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras import utils
from keras.callbacks import ModelCheckpoint,TensorBoard
import numpy as np
import sys

class CustomDataGen():

    def __init__(self, dim_x, dim_y, dim_z, num_class, batch_size):
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.num_class = num_class

    def randomize_ind(self,data):
        indexes = np.arange(len(data))
        np.random.shuffle(indexes)
        return indexes

    def get_data(self,list):

        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size,self.num_class))

        for id, data in enumerate(list):
            im_path = data.split(' ')[0]
            label = int(data.split(' ')[1])
            img = image.load_img(im_path, target_size=(self.dim_x, self.dim_y))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)[0]
            X[id,:,:,:] = x

            y_ = utils.to_categorical(label, self.num_class)
            y[id,...] = y_

        return X, y

    def generate_batch(self, data):

        while 1:
            indexes = self.randomize_ind(data)

            num_batch = int(len(indexes)/self.batch_size)
            for batch_id in range(num_batch):
                temp_list = [data[k] for k in indexes[batch_id*self.batch_size:(batch_id+1)*self.batch_size]]

                X,y = self.get_data(temp_list)

                yield X, y

def read_img_list_from_file(img_dir,file_path):

    data = []
    with open(file_path) as f:
        for line in f:
            data.append(img_dir + line)

    return data

def get_model():
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_conv = vgg.output

    x = Flatten()(last_conv)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation='softmax')(x)

    model = Model(vgg.input, x)

    for layer in model.layers[:19]:
        # print(layer)
        layer.trainable = False

    return model

if __name__ == "__main__":

    num_class = 25
    directory = '../images/'

    train_data = read_img_list_from_file(directory,'../splits/train0.txt')
    test_data = read_img_list_from_file(directory,'../splits/test0.txt')

    batch_size = int(sys.argv[1])
    num_epoch = int(sys.argv[2])

    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    training_generator = CustomDataGen(224, 224, 3, num_class, batch_size).generate_batch(train_data)
    validation_generator = CustomDataGen(224, 224, 3, num_class, batch_size).generate_batch(test_data)

    file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)

    # callbacks_list = [checkpoint, tensorboard]
    callbacks_list = [checkpoint]

    model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(train_data)//batch_size,
                        epochs= num_epoch,
                        validation_data = validation_generator,
                        validation_steps = len(test_data)//batch_size,
                        callbacks=callbacks_list)

    # model.fit_generator(generator=training_generator,
    #                     steps_per_epoch=1,
    #                     epochs=10,
    #                     validation_data=validation_generator,
    #                     validation_steps=1)
