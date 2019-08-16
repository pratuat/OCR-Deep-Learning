#%%
import numpy as np
from os.path import join
import cv2
from keras.preprocessing.image import ImageDataGenerator

#%%

ROOT_DIR = '.'
SRC_DIR = ROOT_DIR

#%%

trainData = np.load(join(SRC_DIR, 'train_data.npy'))

#%%

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.5,
    shear_range=25
)

#%%

augmented_image_data = []
augmented_label_data = []

for index, data in enumerate(trainData):
    print(index)
    aug_iter = datagen.flow(np.array([data[0]]))
    for i in range(100):
        # generate augumented image
        aug_image = next(aug_iter)[0].astype(np.uint8)
        augmented_image_data.append(aug_image/255.)
        
    for i, sigma in enumerate(np.random.normal(25, 10, 30)):
        aug_image = next(aug_iter)[0].astype(np.uint8)
        noisy_img = cv2.GaussianBlur(aug_image, ksize = (3,3), sigmaX = sigma, sigmaY = sigma)
        augmented_image_data.append(noisy_img.reshape((64,64,1))/255.)

    augmented_label_data.extend([data[1:],] * 130)
#%%

np.save('augmented_image_data.npy', augmented_image_data)
np.save('augmented_label_data.npy', augmented_label_data)
