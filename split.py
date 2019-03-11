import pandas as pd
import os
import shutil
df = pd.read_csv('train_labels.csv')

df['id'] = df['id'] + '.tif'

start_directory = './train'
cancerdir = './cancer'
notcancerdir = './notcancer'
cancer = df.loc[df['label'] == 1]
notcancer = df.loc[df['label'] == 0]
fullpath = os.path.join
for dirname, dirnames, filenames in os.walk(start_directory):
    for filename in filenames:
        source = fullpath(dirname, filename)
        if(cancer['id'].str.contains(filename).any()):
            shutil.move(source, fullpath(cancerdir, filename))
        else:
            shutil.move(source, fullpath(notcancerdir, filename))