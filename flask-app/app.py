from flask_ngrok import run_with_ngrok
import os
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
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flask import Flask,render_template, redirect, url_for, request
app = Flask(__name__)
run_with_ngrok(app)  

model_path = os.path.join('snapshots', 'version8_resplit_test_train', 'resnet50_csv_12_inference.h5')
model = models.load_model('/content/aerial_pedestrian_detection/resnet50_csv_12_inference.h5', backbone_name='resnet50')
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
    with open('/content/img_results.txt','w') as f:
      f.write('Number of detections:'+str(count)+'\n')
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    
    file, ext = os.path.splitext(filepath)
    image_name = file.split('/')[-1] + ext
    output_path = os.path.join('/content/flask-app/static/results/', image_name)
    print(output_path)
    draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, draw_conv)

@app.route("/", methods=['GET', 'POST'])
def home():
  sdd_images = os.listdir('/content/aerial_pedestrian_detection/imgs')
  base_path = '/content/aerial_pedestrian_detection/imgs//'
  for image in sdd_images:
    if image!='.ipynb_checkpoints':
      run_detection_image(os.path.join(base_path,image))
  with open('/content/img_results.txt', 'r') as f: 
    return render_template('test.html',text = f.read())
app.run()