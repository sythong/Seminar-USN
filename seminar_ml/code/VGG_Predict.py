import keras
from keras import backend as k
import VGG_MNIST_EVALUATE
from keras.models import load_model
import numpy as np
import cv2
import glob
import os


model_path = 'VGG_MNIST.model'
test_path = '/home/chrisander/datasets/MNIST/predict_test'
image_path = '/home/chrisander/datasets//MNIST/testing/0/10.png'

for name in glob.glob1(test_path, '*.png'):
    path = os.path.join(test_path, name)

    image = cv2.imread(path)
    plot_image = image
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float') / 255.0
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    model = load_model(model_path)
    model.summary()
    model_score = model.predict(image)
    model_score = np.argmax(model_score)
    print('Prediction: ' + str(model_score) + ', Image: ' + str(path))


    plot_image = cv2.resize(plot_image, (400, 400))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Prediction: ' + str(model_score)
    cv2.putText(plot_image, text, (10, 390), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(text, plot_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

del model
k.clear_session()

#  ---------- SINGLE IMAGE PREDICTION ----------
# image = cv2.imread(image_path)
# plot_image = image
# image = cv2.resize(image, (28, 28))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image.astype('float') / 255.0
# image = np.array(image)
# image = np.expand_dims(image, axis=0)

# model = load_model(model_path)
# model_score = model.predict(image)
# model_score = np.argmax(model_score)
# print('Predicted: ' + str(model_score))

# plot_image = cv2.resize(plot_image, (200, 200))
# font = cv2.FONT_HERSHEY_SIMPLEX
# text = 'Predict: ' + str(model_score)
# cv2.putText(plot_image, text, (10, 500), font, 4, (255, 0, 0), 2, cv2.LINE_AA)
# cv2.imshow(text, plot_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# del model
# k.clear_session()