[Article](https://medium.com/@shreykera7/person-detection-from-raspberry-pi-images-40c0bf5d6b8f)

# RPI-Drone-Image-Detection-IOT-Project
Project Members: Shreyas Bhat Kera, N Harishchandra Prasad, Mihir Abhay Deshmukh

## Pipeline

![](IOT-Pipeline.png)

Raspberry Pi:
-          Acquires drone image and corresponding GPS data. (In this project we simulate the images and GPS data, however we have written code to allow data to be captured from the Raspberry Pi Camera Module v2 sensor and the NEO-6M GPS Module sensor.)
-          Authorize and sync the Raspberry Pi to Drive to allow file transfer between the two, using rclone.
-          A cronjob continuously sends any newly acquired data to Drive once a minute.
Google Drive:
-          Storage in the form of two types: drone images and GPS.
-          Store the codes for the model and flask app, as well as the model weights. Code can also be accessed through Github: https://github.com/shreyas-bk/RPI-Drone-Image-Detection-IOT-Project
-          Stores the dataset images from the Stanford Drone Image dataset: https://cvgl.stanford.edu/projects/uav_data/.
Google Colab:
-          Colab gives us access to a GPU instance (the trained model takes around 6 seconds for inference on a CPU, while it takes only 0.3-1 second on the GPU, a 16 GB Tesla T4).
-          We then mount the Drive to access the code, model weights and image and GPS data from the Raspberry Pi. A continuous link to take into account new data is maintained.
-          Colab lets us use necessary Python packages (Keras, Tensorflow, Flask) needed to run the model. A flask app is run to visualize the results on the webpage.
Flask App:
-          Have a real-time dashboard that maintains a stream for new GPS signal and image results obtained from the model, whenever images are captured by the Raspberry Pi.
-          The detections for all the previously captured images are also accessible through a separate link. The GPS data for each of these images is used to generate the corresponding location which is displayed on a map using the Google Maps API. 
-      We use the Flask-Ngrok module in order to provide a public URL rather than connecting to localhost since Colab uses a VM. Since the URL is public, anyone with the URL can access the webpage.
Model Explanation:
-          The model is a pre-trained RetinaNet, trained on a subset of images from the Stanford Drone Dataset: https://github.com/priya-dwivedi/aerial_pedestrian_detection.

