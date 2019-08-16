#%%

import numpy as np
from os import listdir
import re, string, pickle
from os.path import isfile, join
from keras.utils import to_categorical
from PIL import ImageFont, ImageDraw, Image
from sklearn.preprocessing import LabelEncoder

#%%

ROOT_DIR = '.'
DATA_DIR = join(ROOT_DIR, '..', 'data')
SRC_DIR = ROOT_DIR
IMG_DIR = join(ROOT_DIR,'..', 'data', 'images')

#%%

# load font files
font_files = [f for f in listdir(join(DATA_DIR, 'fonts')) if isfile(join(DATA_DIR, 'fonts', f)) and re.search('.(o|t)tf$', f)]

#%%

# image height/width
INPUT_H = INPUT_W = 64

# printable character classes
CHARACTER_CLASSES = string.printable[:-6]

#%%

FONTSIZE_RANGE = [30, 45]

IMAGE_DATA = []

save_image = 0

# flag to save generated character images (.jpg files)

for file in font_files:
    print("[INFO] Generating images for font >>", file)

    file_name, _ = file.split('.')
    font_tags = file_name.split('_')

    for font_size in FONTSIZE_RANGE:
        print("\t[INFO] Generating images for font size >>", font_size)

        # load font file
        font = ImageFont.truetype(join(DATA_DIR, 'fonts', file), font_size)

        for index, character in enumerate(CHARACTER_CLASSES):

            image = Image.new('L', (INPUT_W, INPUT_H), color='white')
            draw = ImageDraw.Draw(image)

            # pre-evaluate font width/height for centering purpose
            w, h = draw.textsize(character, font = font)

            # offset positions for centered font characters
            x_offset = (INPUT_W-w)/2
            y_offset = (INPUT_H-h)/2

            draw.text((x_offset, y_offset), character, font = font)

            if save_image:
                image_path = join(IMG_DIR, '_'.join([file_name, str(index), str(font_size) + '.jpg']))
                image.save(image_path)


            IMAGE_DATA.append([
                file_name,
                font_tags[0],
                character,
                int('B' in font_tags),
                int('I' in font_tags),
                np.array(image).reshape((64,64,1))
            ])

#%%

font_labels = []
char_labels = []
bold_labels = []
italics_labels = []

for data in IMAGE_DATA:
    font, char, bold, italics = data[1:5]
    font_labels.append(font)
    char_labels.append(char)
    bold_labels.append(bold)
    italics_labels.append(italics)

def one_hot_encoder(class_inputs):
    integer_encoder = LabelEncoder()
    encoded_inputs = integer_encoder.fit_transform(class_inputs)
    encoded_inputs = encoded_inputs.reshape(len(encoded_inputs), 1)

    return [integer_encoder, encoded_inputs]

# encode fonts
font_encoder, encoded_fonts = one_hot_encoder(font_labels)

# encode fonts
char_encoder, encoded_char = one_hot_encoder(char_labels)

# encode fonts
bold_encoder, encoded_bold = one_hot_encoder(bold_labels)

# encode fonts
italics_encoder, encoded_italics = one_hot_encoder(italics_labels)


# one hot encode
encoded_font_labels = to_categorical(encoded_fonts)
encoded_character_labels = to_categorical(encoded_char)
encoded_bold_labels = to_categorical(encoded_bold)
encoded_italics_labels = to_categorical(encoded_italics)


TRAIN_LABELS = []

for i in range(len(IMAGE_DATA)):
    TRAIN_LABELS.append([
        IMAGE_DATA[i][5],
        encoded_font_labels[i],
        encoded_character_labels[i],
        encoded_bold_labels[i],
        encoded_italics_labels[i]
    ])

#%%
# dump label_encoder
pickle.dump(
    [font_encoder, char_encoder, bold_encoder, italics_encoder],
    open(join(SRC_DIR, 'label_encoders.p'), 'wb')
)

np.save(join(SRC_DIR, 'train_data.npy'), np.array(TRAIN_LABELS))
print('train_data.npy')

#%%
