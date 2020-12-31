import numpy as np
import cv2
import time
import tensorflow as tf
import os

from fen_chess_data.utils import distance, find_max_contour_area, find_outer_corners, do_perspective_transform, split_chessboard, preds_to_fen, generate_img

weights_fname   = f"YOLO_files/yolov4-tiny-obj_best.weights"  # substitute your weights
cfg_fname       = f"YOLO_files/yolov4-tiny-obj.cfg"           # substitute your cfg
classes_fname   = f"YOLO_files/obj.names"                     # substitute your class names

model = tf.keras.models.load_model('fen_chess_data/models/model_best.h5')

# Load Yolo
net = cv2.dnn.readNet(
    weights_fname, 
    cfg_fname
)

# enable GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open(classes_fname, "r") as f: classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading camera
address = 'http://192.168.1.75:8080/video'

cap = cv2.VideoCapture(address)
# cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()

frame_id = 0
wh_adjust = 1.20
confidence_thr = 0.2

while True:

    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    clip = min(height, width)
    frame = frame[:clip, :clip, :]
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

    # save original copy of frame (in color)
    frame_orig = frame.copy()

    # for frame analysis
    cv2.imwrite(f'fen_chess_data/data/test_img.jpg', frame_orig)

    d = 128
    frame = cv2.resize(frame, (d, d), interpolation = cv2.INTER_AREA)
    height, width, channels = frame.shape

    frame = cv2.Canny(frame, width, height)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (height, width), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    centers = []

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_thr:

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width * wh_adjust)
                h = int(detection[3] * height * wh_adjust)

                # change w & h to max(w, h) to make it square always
                w = h = max(w, h)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                centers.append([center_x, center_y])
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    # find most central box
    mid_x, mid_y, min_dst = width // 2, height // 2, float('inf')
    i = None

    for j in range(len(centers)):

        if j in indexes:

            center_x, center_y = centers[j]
            dst = distance([center_x, center_y], [mid_x, mid_y])

            if dst < min_dst:
                min_dst = dst
                i = j

    if i:
        x, y, w, h = boxes[i]

        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]

        dim = frame.shape[0]
        orig_dim = frame_orig.shape[0]

        x_orig = int( np.round( orig_dim * ( x / dim ) ) )
        y_orig = int( np.round( orig_dim * ( y / dim ) ) )
        w_orig = int( np.round( orig_dim * ( w / dim ) ) )
        h_orig = int( np.round( orig_dim * ( h / dim ) ) )

        # for frame analysis
        with open('fen_chess_data/data/test_img_yolo_box.txt', 'w') as f:
            f.write( ' '.join([str(x_orig), str(y_orig), str(w_orig), str(h_orig)]) )
            f.close()

        # make copy of img for later
        img = frame_orig[ y_orig : y_orig + h_orig, x_orig : x_orig + w_orig ].copy()

        # rectangle text
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + 5), color, -1)
        cv2.putText(frame, str(round(confidence, 2)), (x, y + 5), font, 1, (255,255,255), 1)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    # do perspective shift, display in 2nd window
    if i and img.shape[0] > 0 and img.shape[1] > 0:

        w = img.shape[0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = find_max_contour_area(contours)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        c = contours[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        pts = find_outer_corners(img, approx)

        img_orig = frame_orig[ y_orig : y_orig + h_orig, x_orig : x_orig + w_orig ].copy()
        img_orig = do_perspective_transform(img_orig, pts)

        # save original copy for piece prediction
        img_orig_predict = img_orig.copy()

        img_orig = cv2.resize(img_orig, (512, 512), interpolation = cv2.INTER_AREA)

        w, h, _ = img_orig.shape
        dims = list(range(0, w + 1, w // 8))

        for i in dims:
            img_orig = cv2.line(img_orig, (i, 0), (i, w), (255,0,0), 2)
            img_orig = cv2.line(img_orig, (0, i), (w, i), (255,0,0), 2)

        cv2.imshow("img", img_orig)

        for p in pts:
            p[0] += x_orig
            p[1] += y_orig
            cv2.circle(frame_orig, (p[0], p[1]), 3, (0,0,255), -1)

        for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            frame_orig = cv2.line(frame_orig, (pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]), (0,0,255), 1)

    frame_orig = cv2.resize(frame_orig, (1024, 1024), interpolation = cv2.INTER_AREA)
    cv2.imshow("frame_orig", frame_orig)

    key = cv2.waitKey(1)
    if key == 27: break

    if key == 13:

        # predict class of each 64 squares
        img_orig = cv2.resize(img_orig_predict, (256, 256), interpolation = cv2.INTER_AREA)

        # convert to B&W but with shape (w, h , 3) for model compatibility
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)

        imgs = split_chessboard(img_orig)
        preds = model.predict( np.float32(np.array(imgs)) )

        # save imgs to examine
        cv2.imwrite(f'fen_chess_data/predictions/im_orig.jpg', img_orig)
        for i, im in enumerate(imgs):
            cv2.imwrite(f'fen_chess_data/predictions/im_{i}.jpg', im)

        fen = preds_to_fen(preds)

        final_img = generate_img(fen, large_dim=400)

        # visualize final board
        cv2.imshow("final_img", final_img)

        # another while loop so images don't disappear
        while True:

            key2 = cv2.waitKey(1)
            if key2 == 13:
                break

        break

cap.release()
cv2.destroyAllWindows()
