# -*- coding: utf-8 -*-
from os.path import join
from tensorflow.python.keras.models import load_model, save_model

#%%

ROOT_DIR = '.'
MODEL_DIR = join(ROOT_DIR, '..', 'data_out', 'model')
#%%

def saveModel(model, tag='1'):
    save_model(model, join(MODEL_DIR, '%s.h5' % tag), include_optimizer = True, overwrite = True)
    return tag
    
def loadModel(tag = '1', trainable = False):
    model = load_model(filepath = join(MODEL_DIR, '%s.h5' % tag), compile = False)

    print("Loaded model from disk")
    
    if not trainable:
        for layer in model.layers:
            layer.trainable = False
    
    return model
