import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Activation, Add, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.activations import relu
import tensorflow as tf
import time

print('getting model...')

def ResidualBlock(inputs, filters):
    x = Conv2D(filters, (3,3), strides=1, padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(filters, (3,3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, inputs])
    return x

def DeConv2D(inputs):
    x = UpSampling2D(size=2)(inputs)
    x = Conv2D(256, (3,3), strides=1, padding='same')(x)
    x = Activation('relu')(x)
    return x

inputs = Input(shape=(None,None,3))
inner = inputs

inner1 = Conv2D(64, (9,9), strides=1, padding='same')(inner)
inner1 = Activation('relu')(inner1)

inner = inner1
inner = ResidualBlock(inner, 64)

for _ in range(15):
    inner = ResidualBlock(inner, 64)

inner = Conv2D(64, (3,3), strides=1, padding='same')(inner)
inner = BatchNormalization(momentum=0.8)(inner)
inner = Add()([inner, inner1])

inner = DeConv2D(inner)

outputs = Conv2D(3, (9,9), strides=1, padding='same', activation='sigmoid')(inner)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='mse', optimizer='Adam')

model.load_weights('generator-19.h5') #17/19

print('enhancing images...')

for img_name in range(1, 17):
    start = time.time()
    
    img_name = 'image' + str(img_name)
    print(img_name, end='\t')
    
    img = 'images/' + img_name + '_lowres.png'

    img = load_img(img)

    size = img.size

    img = img_to_array(img) / 255
    
    img = np.array([img])

    img = model.predict(img)[0]

    #img = relu(img) # for old tanh models

    img = np.array(img) * 255

    img = array_to_img(img)

    img.save('images/' + img_name + '_highres.png')

    print(str(round(time.time()-start, 2)) + 's')
