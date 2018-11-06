
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import load_model
import numpy as np
k = keras.backend


def EvaluatingModel():

    # Path to desired model
    model_path = 'VGG_MNIST.model'

    # Path to Data-set
    test_data_dir = '/home/chrisander/datasets/MNIST/testing'

    # Batch size
    batch_size_test = 32

    # Dimension of images / Input shape
    img_width, img_height = 28, 28

    # RGB channel converted to float between 0 and 1
    test_gen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_gen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size_test,
        shuffle=False,
        class_mode='categorical')

    num_test_batches = np.ceil(test_generator.samples // batch_size_test)

    model = load_model(model_path)
    model_score = model.evaluate_generator(test_generator, steps=num_test_batches)
    print('Evaluating model')
    print(model.metrics_names)
    print(model_score)
    print('Model scored ' + str(model_score[1]*100) + '% accuracy')
    k.clear_session()


