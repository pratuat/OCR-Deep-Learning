import sys, csv
import pickle
import numpy as np
from os.path import join
from persistent_models import loadModel
from sklearn.preprocessing import LabelEncoder
#%%

ROOT_DIR = '.'
SRC_DIR = ROOT_DIR
MODEL_DIR = join(ROOT_DIR, '..', 'data_out', 'model')
OUTPUT_DIR = join(ROOT_DIR, '..', 'data_out', 'test')
#%%

_, input_file, output_file, *args = sys.argv

# load test data
test_data = np.load(input_file)

# load model
model = loadModel(tag='char_font_bold_italics', trainable = False)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# load encoders
font_encoder, char_encoder, bold_encoder, italic_encoder = pickle.load(open(join(SRC_DIR, 'label_encoders.p'), 'rb'))
#%%

predictions = model.predict(test_data, batch_size = 20, verbose = 1)

char_predictions = char_encoder.inverse_transform([np.argmax(x) for x in predictions[0]])
font_predictions = font_encoder.inverse_transform([np.argmax(x) for x in predictions[1]])
bold_predictions = bold_encoder.inverse_transform([np.argmax(x) for x in predictions[2]])
italics_predictions = italic_encoder.inverse_transform([np.argmax(x) for x in predictions[3]])

#%%

with open(join(OUTPUT_DIR, output_file + '.csv') , 'w') as file:
  writer = csv.writer(file, delimiter = ',', quoting=csv.QUOTE_MINIMAL)

  for i in range(len(test_data)):
    writer.writerow([char_predictions[i], font_predictions[i], bold_predictions[i], italics_predictions[i]])

  file.close()
