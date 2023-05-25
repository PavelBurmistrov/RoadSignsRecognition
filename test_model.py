import os
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from create_model import TSR


class ModelTester:

    def __init__(self, testing_model):
        # Classes of traffic signs
        self.__classes = {0: 'Speed limit (20km/h)',
                          1: 'Speed limit (30km/h)',
                          2: 'Speed limit (50km/h)',
                          3: 'Speed limit (60km/h)',
                          4: 'Speed limit (70km/h)',
                          5: 'Speed limit (80km/h)',
                          6: 'End of speed limit (80km/h)',
                          7: 'Speed limit (100km/h)',
                          8: 'Speed limit (120km/h)',
                          9: 'No passing',
                          10: 'No passing veh over 3.5 tons',
                          11: 'Right-of-way at intersection',
                          12: 'Priority road',
                          13: 'Yield',
                          14: 'Stop',
                          15: 'No vehicles',
                          16: 'Veh > 3.5 tons prohibited',
                          17: 'No entry',
                          18: 'General caution',
                          19: 'Dangerous curve left',
                          20: 'Dangerous curve right',
                          21: 'Double curve',
                          22: 'Bumpy road',
                          23: 'Slippery road',
                          24: 'Road narrows on the right',
                          25: 'Road work',
                          26: 'Traffic signals',
                          27: 'Pedestrians',
                          28: 'Children crossing',
                          29: 'Bicycles crossing',
                          30: 'Beware of ice/snow',
                          31: 'Wild animals crossing',
                          32: 'End speed + passing limits',
                          33: 'Turn right ahead',
                          34: 'Turn left ahead',
                          35: 'Ahead only',
                          36: 'Go straight or right',
                          37: 'Go straight or left',
                          38: 'Keep right',
                          39: 'Keep left',
                          40: 'Roundabout mandatory',
                          41: 'End of no passing',
                          42: 'End no passing veh > 3.5 tons'}
        self.testing_model = testing_model

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, new_classes):
        self.__classes = new_classes

    def test_on_img(self, img, model):
        data = []
        image = Image.open(img)

        image = image.resize((30, 30))
        data.append(np.array(image))

        X_test = np.array(data)
        predict_x = model.predict(X_test)
        Y_pred = np.argmax(predict_x, axis=1)
        return Y_pred


def main():
    tsr_model = TSR()

    if os.path.isfile(tsr_model.MODEL_PATH):
        tsr_model.load_model(tsr_model.MODEL_PATH)
        tester = ModelTester(tsr_model)
        prediction = tester.test_on_img(os.path.join(tester.testing_model.CUR_PATH, r'Train\0\00000_00000_00000.png'),
                                        tester.testing_model.model)
        traffic_sign_name = tester.classes[int(prediction)]
        print("Predicted traffic sign is: ", traffic_sign_name)
    else:
        print("Try to create the model before using it !!!")


if __name__ == "__main__":
    main()
