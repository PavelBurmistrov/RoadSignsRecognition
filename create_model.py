from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score


class Model(ABC):

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def build_model_architecture(self, X_train):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def save_model(self, model_path):
        pass

    @abstractmethod
    def test_model(self):
        pass

    def show_training_accuracy_graph(self, history):
        # accuracy
        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    def show_training_loss_graph(self, history):
        # Loss
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


class TSR(Model):
    __CLASSES_NUM = 43
    __CUR_PATH = os.getcwd()
    __SAVE_PATH_DIR = os.path.join(__CUR_PATH, 'save_data')
    __MODEL_PATH = os.path.join(__SAVE_PATH_DIR, 'TSR.h5')
    __TRAIN_DATA_DIR = os.path.join(__CUR_PATH, 'Train')
    __TEST_DATA = os.path.join(__CUR_PATH, 'Test.csv')
    __EPOCHS = 20

    def __init__(self):
        self.__model = Sequential()

    @property
    def MODEL_PATH(self):
        return self.__MODEL_PATH

    @property
    def CUR_PATH(self):
        return self.__CUR_PATH

    @property
    def SAVE_PATH_DIR(self):
        return self.__SAVE_PATH_DIR

    @property
    def model(self):
        return self.__model

    def train_data_transformation(self, train_path):
        data = []
        labels = []

        for i in range(TSR.__CLASSES_NUM):
            path = os.path.join(train_path, str(i))
            images = os.listdir(path)
            for a in images:
                try:
                    image = Image.open(path + '\\' + a)
                    image = image.resize((30, 30))
                    image = np.array(image)
                    data.append(image)
                    labels.append(i)
                except Exception as e:
                    print(e)

        data = np.array(data)
        labels = np.array(labels)

        return data, labels

    def prepare_data(self):
        data = []
        labels = []

        if not os.path.isdir(TSR.__SAVE_PATH_DIR):
            os.mkdir(TSR.__SAVE_PATH_DIR)

        if os.path.isfile(os.path.join(TSR.__SAVE_PATH_DIR, 'data.npy')) \
                and os.path.isfile(os.path.join(TSR.__SAVE_PATH_DIR, 'target.npy')):
            data = np.load(os.path.join(TSR.__SAVE_PATH_DIR, 'data.npy'))
            labels = np.load(os.path.join(TSR.__SAVE_PATH_DIR, 'target.npy'))
        else:
            try:
                if not os.path.isdir(TSR.__TRAIN_DATA_DIR):
                    raise Exception("Error! No actual train data for model studying in the project!")
                data, labels = self.train_data_transformation(TSR.__TRAIN_DATA_DIR)

                print(data)
                print(labels)

                np.save(os.path.join(TSR.__SAVE_PATH_DIR, 'data'), data)
                np.save(os.path.join(TSR.__SAVE_PATH_DIR, 'target'), labels)
            except Exception as e:
                print(e)

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def build_model_architecture(self, X_train):
        self.__model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
        self.__model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        self.__model.add(MaxPool2D(pool_size=(2, 2)))
        self.__model.add(Dropout(rate=0.25))
        self.__model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.__model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.__model.add(MaxPool2D(pool_size=(2, 2)))
        self.__model.add(Dropout(rate=0.25))
        self.__model.add(Flatten())
        self.__model.add(Dense(256, activation='relu'))
        self.__model.add(Dropout(rate=0.5))
        # We have 43 classes that's why we have defined 43 in the dense
        self.__model.add(Dense(43, activation='softmax'))

        # Compilation of the model
        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        y_train = to_categorical(y_train, 43)
        y_test = to_categorical(y_test, 43)

        if not os.path.isfile(TSR.__MODEL_PATH):
            self.build_model_architecture(X_train)
            history = self.__model.fit(X_train, y_train, batch_size=32, epochs=TSR.__EPOCHS,
                                       validation_data=(X_test, y_test))

            self.show_training_accuracy_graph(history)
            self.show_training_loss_graph(history)
            self.test_model()
            # Save
            self.__model.save(TSR.__MODEL_PATH)
        else:
            self.__model = load_model(TSR.__MODEL_PATH)

    def load_model(self, model_path):
        self.__model = load_model(model_path)

    def save_model(self, model_path):
        self.__model.save(model_path)

    def test_data_transformation(self, test_data_path):
        y_test = pd.read_csv(test_data_path)
        label = y_test["ClassId"].values
        imgs = y_test["Path"].values
        data = []
        for img in imgs:
            image = Image.open(img)
            image = image.resize((30, 30))
            data.append(np.array(image))
        X_test = np.array(data)
        return X_test, label

    def test_model(self):
        X_test, label = self.test_data_transformation(TSR.__TEST_DATA)

        predict_x = self.__model.predict(X_test)
        Y_pred = np.argmax(predict_x, axis=1)

        print("\nY_pred:\n", Y_pred)

        print("\nlabel:\n", label)

        print("\nAccuracy score is ", accuracy_score(label, Y_pred))


def main():
    tsr_model = TSR()
    tsr_model.train_model()



if __name__ == "__main__":
    main()