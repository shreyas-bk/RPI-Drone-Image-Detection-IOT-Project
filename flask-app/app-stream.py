from flask_ngrok import run_with_ngrok
import os
import sys
from stat import S_ISREG, ST_CTIME, ST_MODE
from pathlib import Path
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf
import itertools
import time
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flask import Flask,render_template, redirect, url_for, request,Response
app = Flask(__name__)
run_with_ngrok(app)  

model_path = os.path.join('snapshots', 'version8_resplit_test_train', 'resnet50_csv_12_inference.h5')
model = models.load_model('/content/RPI-Drone-Image-Detection-IOT-Project/resnet50_csv_12_inference.h5', backbone_name='resnet50')
labels_to_names = {0: 'Biker', 1: 'Car', 2: 'Bus', 3: 'Cart', 4: 'Skater', 5: 'Pedestrian'}

def run_detection_image(filepath):
    image = read_image_bgr(filepath)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    print("processing time: ", time.time() - start)
    boxes /= scale
    count = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        if label!=5 and label!=0:
          continue
        count+=1
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)
        draw_caption(draw, b, caption)
    with open('/content/RPI-Drone-Image-Detection-IOT-Project/flask-app/static/results/img_results.txt','w') as f:
      f.write('Number of detections:'+str(count)+'\n')
    
    file, ext = os.path.splitext(filepath)
    image_name = file.split('/')[-1] + ext
    output_path = os.path.join('/content/RPI-Drone-Image-Detection-IOT-Project/flask-app/static/results', image_name)
    draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, draw_conv)

def get_chrono(dir_path):
  entries = (os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path))
  entries = ((os.stat(path), path) for path in entries)
  entries = ((stat[ST_CTIME], path)
           for stat, path in entries if S_ISREG(stat[ST_MODE]))
  entries_list = []
  for entry in entries:
    entries_list.append(entry)
  sorted_entries_list = sorted(entries_list)
  sorted_paths = []
  for sorted_entry in sorted_entries_list:
    sorted_paths.append(sorted_entry[1])
  return sorted_paths

def gen_frames(): 
    prev_imgs = [] 
    prev_latest_image = ''
    dirpath = '/content/gdrive/MyDrive/images'
    while True:
        #print(prev_latest_image)
        posix_images = get_chrono(dirpath)
        drone_imgs = []
        for image in posix_images:
          if '.ipynb_checkpoints' not in str(image) and 'shortcut-targets-by-id' not in str(image):
            drone_imgs.append(image)
        if len(drone_imgs)!=len(prev_imgs):
              new_imgs = drone_imgs[len(drone_imgs)-(len(drone_imgs)-len(prev_imgs)):]
              latest_image = drone_imgs[-1]
              print(new_imgs,latest_image)
        #if prev_latest_image!=latest_image:
              if '.ipynb_checkpoints' not in new_imgs:
                  for image_to_run in new_imgs:
                      run_detection_image(image_to_run)
                  prev_latest_image = latest_image
                  prev_imgs = drone_imgs
        img_path = 'static/results/'+str(prev_latest_image)[len(dirpath):]
        buff = cv2.imread('/content/RPI-Drone-Image-Detection-IOT-Project/flask-app/'+img_path)
        (flag, buff) = cv2.imencode(".jpg", buff)
        frame = buff.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_feed')
def count_feed():
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            while True:
                with open('/content/RPI-Drone-Image-Detection-IOT-Project/flask-app/static/results/img_results.txt','r') as f:
                    text = f.read()
                yield "data: %s \n\n" % (text)
                time.sleep(1)  # an artificial delay
        return Response(events(), content_type='text/event-stream')

@app.route("/", methods=['GET', 'POST'])
def home():
  
  return render_template('image_stream.html')
with open('/content/RPI-Drone-Image-Detection-IOT-Project/flask-app/static/results/img_results.txt','w') as f:
      f.write('Number of detections:\n')
app.run()
