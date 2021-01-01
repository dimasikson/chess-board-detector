# chess_computer_vision_project

# 1. What is this?

This project is an end-to-end computer vision pipeline to detect 2d chess boards and output the FEN. Works on CPU in real time. Below are the pipeline steps:

![](https://github.com/dimasikson/chess_computer_vision_project/blob/master/chess%20CV%20project%20-%20pipeline%20overview.png?raw=true)
#### Chart 1. High-level pipeline overview.

### Demo video: https://www.youtube.com/watch?v=JC1B7B19dEQ
![](https://media.giphy.com/media/8TurMGdUisMdJmPxqz/giphy.gif)

# 2. Detailed pipeline overview

## 2.1 YOLO object detection

We use openCV to evaluate frames in real time, either from a webcam or phone footage via an IP. This was designed to deploy on an iOS app (TBD).

![](https://i.gyazo.com/2f81c3a83d517a065e2b5ef5612b2775.png)
#### Chart 2. Original frame.

We use the tiny YOLO from https://github.com/AlexeyAB/darknet for object detection. For preprocessing, I found that applying the Canny edge detection improves the YOLO performance, particularly the recall. https://docs.opencv.org/master/da/d22/tutorial_py_canny.html

![](https://i.gyazo.com/24c8e5c6cfc1bbfb54113f23b92144b8.png)
#### Chart 3. Downscaled image + applied Canny edge detection
#### Chart 4. YOLO prediction applied (only the most central prediction taken)

## 2.2 Corner detection + perspective shift

Once we have the YOLO object bounding box, I have found the following pipeline to be the most reliable way to find the corners:
1. Adaptive threshold, cv2.adaptiveThreshold() 
2. Find contours, cv2.findContours()
3. Select contour with largest area, cv2.contourArea()
4. Find corners of that contour, cv2.arcLength() -> cv2.approxPolyDP()

Finally, we apply perspective shift on the found corners to line up each 64 squares with the fixed grid.

Below this pipeline is visualized:

![](https://i.gyazo.com/e4fda35f8602ef234e94c62bfe9c8efa.png)
#### Chart 5. Original image in YOLO box
#### Chart 6. Adaptive threshold applied

![](https://i.gyazo.com/426ead6b3a999c45949f17b69832b4d8.png)
#### Chart 7. Find contours function applied

![](https://i.gyazo.com/a02017caec76a8623237cd0e0a9ea352.png)
#### Chart 8. Selected contour with largest area + found most outer points

![](https://i.gyazo.com/0be1370adf0c488e4767dd6ddf2679c0.png)
#### Chart 9. Corners visualized on original YOLO image
#### Chart 10. Perspective transform applied

## 2.3 Image classification for chess pieces

Once we have the board lined up with the grid, we split the image into 64 squares and classify each square using a small CNN.

The model was trained on generated images. I used piece and board templates from https://github.com/koryakinp/chess-generator and positions from https://www.kaggle.com/ronakbadhe/chess-evaluations. To augment the data, I also added random perspective shifting in order to simulate imprecise corner detection. To recreate, use the generate_imgs.ipynb notebook and follow instructions in section 6.

The model uses input of shape (32, 32, 3) and output of shape (13) for the 13 square classes (6 pieces x 2 colors + 1 for empty squares). The layers are 2d convolutional layers with batchNorm and ReLU activation.

![](https://i.gyazo.com/762ea59f2cf6273bc01bfee06aa535cb.png)
#### Chart 11. Example square to classify
#### Chart 12. Prediction probabilities of above example

![](https://i.gyazo.com/6ca6b8c8b2a6be0a3335653ded2d5351.png)
#### Chart 13. Example output

![](https://i.gyazo.com/acc898cd0fc70180a5e1fb4d63225bae.png)
#### Chart 14. CNN square classifier model summary

# 3. Performance and areas to improve

## 3.1 YOLO performance

The YOLO model performs reasonably fast on CPU (~15 FPS) and accurately (100% mAP on train & test, pretty good on live validation footage also). I would most want to improve the recall, as it's usually better to have a larger box as the corner detection can usually handle an imprecise YOLO box. 

I found more training images not necessarily improving the YOLO performance, but removing low quality images improving the performance a lot. Perhaps more curation of the dataset could be done. 

## 3.2 CNN piece classifier performance

The train & test square classification accuracy is 99.5%. This suggests that around 72.6% of board captures will have all 64 squares correctly classified, because `0.995 ** 64 = 0.726`

On live validation data, I found the roughly 1 / 5 images would have all 64 squares correct, which is pretty bad. The puts validation accuracy closer to 97.5%, because `0.2 ** (1/64) = 0.975`

Frequent faults of the model would be:
- Classifying a white piece as a black piece of the correct piece type, particularly the Queen and Rook, particularly in low light
- Classifying a black Bishop as a black Pawn

More work on data augmentation needs to be done, particularly taking normalizing for light condition.

Another way to generate training data in an efficient way could be to screencap the output of corner detection, in order to line up the training data with real world validation data, but this felt like it would make this pipeline even more narrow than it already is.

# 4. Instructions to run

### 0. install requirements

### 1. pass your camera / webcam into cv2.VideoCapture in yolov4_tiny_CPU.py

uncomment line 36 for webcam use OR change line 33 to your phone camera's IP address. I was using this app https://apps.apple.com/us/app/ip-webcam-home-security-camera/id1264454867. It's pretty bad but it got the job done for me.

### 2. run 'python yolov4_tiny_CPU.py' in command line

Press `Enter` to evaluate image and `Esc` to close

# 5. Instructions to retrain YOLO

### 0. [optional] to label your own images, git clone https://github.com/tzutalin/labelImg and follow instructions to run

You can use the video_preprocess_yolo.ipynb notebook to help with preprocessing images. I used resize + Canny for chessboards, more detail in the pipeline explanation above.

### 1. git clone https://github.com/AlexeyAB/darknet and follow instructions to compile & train custom objects

For GPU training, make sure to use the darknet.exe in vcpkg\installed\x64-windows\tools\darknet

# 6. Instructions to train CNN

### 0. cd fen_chess_data, git clone https://github.com/koryakinp/chess-generator

### 1. generate images using generate_imgs.ipynb notebook

### 2. run 'python train.py'
