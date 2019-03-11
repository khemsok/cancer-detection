import os
from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
import tensorflow as tf
import flask
import numpy as np
from keras.preprocessing import image
import pandas as pd

app = Flask(__name__)
model = load_model('model2.h5')
graph = tf.get_default_graph()
prediction = 'yo'

APP_ROOT = os.path.dirname(os.path.abspath('__file__'))

df = pd.read_csv('train_labels.csv')
df['id'] = df['id'] + '.tif'

def checkAccuracy(arr_pred, og_filename):
    x = 0
    arr_pred2 = []
    for i in arr_pred:
        if(i == 'CANCER'):
            arr_pred2.append(1)
        else:
            arr_pred2.append(0)
        x+=1
    x = 0
    correct = 0
    # print(og_filename)
    for i in og_filename:
        # print(i)
        # print(df[df['id']==i])
        check = df[df['id'] == i]
        # print('CHECK:', check.values[0][1], '--- ACTUAL', arr_pred2[x])
        if len(check) != 0:
            if check.values[0][1] == arr_pred2[x]:
                correct+=1
            x+=1
        else: return 0
    return correct/len(arr_pred)


def imgProcessing(imgdir):
    test_image = image.load_img(imgdir, target_size=(96, 96))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    with graph.as_default():
        result = model.predict(test_image)
        if result[0][0] < 0.5:
            return 'cancer'
        else:
            return 'not cancer'

@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():
    target = os.path.join(APP_ROOT, 'images/')
    if not os.path.isdir(target):
        os.mkdir(target)
    count = 0
    dest_list = []
    og_filename = []
    for file in request.files.getlist('file'):
        og_filename.append(file.filename)
        filename = 'test' + str(count) + '.tif'
        print(filename)
        destination = '/'.join([target, filename])
        file.save(destination)
        dest_list.append(destination)
        count += 1
    arr_pred = []
    for i in range(count):
        imgdir = './images/test'+str(i)+'.tif'
        arr_pred.append(imgProcessing(imgdir).upper())
    print(arr_pred)
    print('ACCURACY',checkAccuracy(arr_pred, og_filename))
    acc = checkAccuracy(arr_pred, og_filename)
    countcancer = 0
    countnotcancer = 0

    for i in arr_pred:
        if(i == 'CANCER'):
            countcancer += 1
        else:
            countnotcancer += 1
    return render_template('index2.html', arr_pred = arr_pred, len = len(arr_pred), count1 = countcancer, count2 = countnotcancer, avg = acc)

    # return render_template('index2.html', arr_pred = arr_pred, len = len(arr_pred), count1 = countcancer, count2 = countnotcancer, avg1 = countcancer/len(arr_pred), avg2 = countnotcancer/len(arr_pred))


app.run()
