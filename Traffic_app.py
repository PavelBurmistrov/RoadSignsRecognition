from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
from create_model import TSR
from test_model import ModelTester

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        tsr_model = TSR()
        tsr_model.train_model()

        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)

        received_images_path_dir = os.path.join(tsr_model.SAVE_PATH_DIR, 'received_images')

        if not os.path.isdir(received_images_path_dir):
            os.mkdir(received_images_path_dir)

        user_image_save_path = os.path.join(received_images_path_dir, file_path)

        f.save(user_image_save_path)

        # Make prediction
        tester = ModelTester(tsr_model)
        prediction = tester.test_on_img(
            user_image_save_path,
            tester.testing_model.model)
        traffic_sign_name = tester.classes[int(prediction)]
        result = "Полученный прогноз🚦: " + traffic_sign_name

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
