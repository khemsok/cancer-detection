import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd

model = load_model('model2.h5')

def imgProcessing(imgdir):
    test_image = image.load_img(imgdir, target_size=(96, 96))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] < 0.5:
        return 1
    else:
        return 0



df = pd.read_csv('./sample_submission.csv')

df['id'] = df['id']+'.tif'

for i in range(len(df)):
    imgdir = './test/' + df['id'][i]
    df['label'][i] = imgProcessing(imgdir)
    print(i,'----', df['label'][i])

df['id'] = df['id'].str.replace('.tif', '')


export_csv = df.to_csv(r"C:\Users\actzl\OneDrive\Desktop\Work\histopathologic-cancer-detection\submission\sample_submission.csv",index=False)