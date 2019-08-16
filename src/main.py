# -*- coding: utf-8 -*-
#%%
import numpy as np
from os.path import join

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, concatenate

from persistent_models import saveModel, loadModel
#%%

ROOT_DIR = '.'
SRC_DIR = ROOT_DIR
IMAGE_SIZE = 64
N_EPOCHS = 6
#%%

input_images = np.load(join(SRC_DIR, 'augmented_image_data.npy'))
label_data = np.load(join(SRC_DIR, 'augmented_label_data.npy'))

output_fonts = []
output_chars = []
output_bold = []
output_italics = []

for data in label_data:
    output_fonts.append(data[0])
    output_chars.append(data[1])
    output_bold.append(data[2])
    output_italics.append(data[3])

output_font_data = np.array(output_fonts)
output_char_data = np.array(output_chars)
output_bold_data = np.array(output_bold)
output_italic_data = np.array(output_italics)

#%%

### TRAINING CHARACTER CLASSIFIER (VGG-16 INSPIRED)
input_layer = Input(shape=(64,64,1), name = 'input_layer')

net = Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation='relu', name='CH_1_CONV_1')(input_layer)
net = Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation='relu', name='CH_2_CONV_2')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='CH_3_MP_1')(net)

net = Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='CH_4_CONV_3')(net)
net = Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='CH_5_CONV_4')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='CH_6_MP_2')(net)

net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu', name='CH_7_CONV_5')(net)
net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu', name='CH_8_CONV_6')(net)
net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation='relu', name='CH_9_CONV_7')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='CH_10_MP_3')(net)
net = Flatten(name='CH_11_FLT_1')(net)

net = Dense(400, activation='relu', name='CH_12_DNS_1')(net)
net = Dense(400, activation='relu', name='CH_13_DNS_2')(net)
net = Dense(94, activation='softmax', name='output_char')(net)

output_layer = net

model_1 = Model(inputs=input_layer, outputs=output_layer)
model_1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model_1.summary())

#%%

model_1.fit(x = input_images, y = [output_char_data], epochs = N_EPOCHS, batch_size = 94, shuffle=True, validation_split=0.2)

#%%

saveModel(model_1, tag = 'char')


#%%

char_model = loadModel(tag = 'char', trainable = False)

input_layer = char_model.inputs[0]
output_char = char_model.outputs[0]

aux_tensor_1 = char_model.get_layer('CH_11_FLT_1').output

net = Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation='relu', name='FT_1_CONV_1')(input_layer)
net = MaxPooling2D(pool_size=2, strides=2, name='FT_2_MP_1')(net)
net = Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation='relu', name='FT_3_CONV_2')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='FT_4_MP_2')(net)

net = Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='FT_5_CONV_3')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='FT_6_MP_3')(net)
net = Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='FT_7_CONV_4')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='FT_8_MP_4')(net)

net = Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='relu', name='FT_9_CONV_5')(net)
net = MaxPooling2D(pool_size=2, strides=2, name='FT_10_MP_5')(net)
net = Flatten(name='FT_11_FLT_1')(net)

net = concatenate([net, aux_tensor_1], name = 'FT_12_CONCAT_1')
net = Dense(200, activation='relu', name='FT_13_DNS_1')(net)
net = Dense(200, activation='relu', name='FT_14_DNS_2')(net)
output_font = Dense(11, activation='softmax', name='output_font')(net)

model_2 = Model(inputs=input_layer, outputs=[output_char, output_font])
model_2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model_2.summary())

#%%

model_2.fit(x = input_images, y = [output_char_data, output_font_data], epochs = N_EPOCHS, batch_size = 94, shuffle=True, validation_split=0.2)

#%%

saveModel(model_2, tag = 'char_font')


#%%

char_font_model = loadModel(tag = 'char_font', trainable = False)

input_layer = char_font_model.inputs[0]
output_char, output_font = char_font_model.outputs

aux_tensor_1 = char_font_model.get_layer('FT_12_CONCAT_1').output

net = Dense(100, activation='relu', name='B_1_DNS_1')(aux_tensor_1)
net = Dense(100, activation='relu', name='B_2_DNS_2')(net)
output_bold = Dense(2, activation='softmax', name='output_bold')(net)

net = Dense(200, activation='relu', name='I_1_DNS_1')(aux_tensor_1)
net = Dense(200, activation='relu', name='I_2_DNS_2')(net)
output_italics = Dense(2, activation='softmax', name='output_italics')(net)

model_3 = Model(inputs=input_layer, outputs=[output_char, output_font, output_bold, output_italics])

model_3.compile(
    optimizer='rmsprop',
    loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    metrics=['accuracy']
)

print(model_3.summary())

#%%

model_3.fit(x = input_images, y = [output_char_data, output_font_data, output_bold_data, output_italic_data], epochs = N_EPOCHS, batch_size = 94, shuffle=True, validation_split=0.2)

#%%

saveModel(model_3, tag = 'model')

