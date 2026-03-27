import os
import cv2
import time
import numpy as np
import torch
from torch.autograd import Variable
from models.factory import build_net

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    image = image[[2, 1, 0], :, :]
    return image

def detect_face(net, img, tmp_shrink):
    image = cv2.resize(img, None, None, fx=tmp_shrink, fy=tmp_shrink, interpolation=cv2.INTER_LINEAR)
    x = to_chw_bgr(image)
    x = x.astype('float32') / 255.
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net.test_forward(x)[0]
    detections = y.data.cpu().numpy()
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    boxes = []
    scores = []
    for i in range(detections.shape[1]):
        j = 0
        while ((j < detections.shape[2]) and detections[0, i, j, 0] > 0.0):
            pt = (detections[0, i, j, 1:] * scale)
            score = detections[0, i, j, 0]
            boxes.append([pt[0], pt[1], pt[2], pt[3]])
            scores.append(score)
            j += 1

    if len(boxes) == 0:
        return np.array([])

    boxes = np.array(boxes)
    det_conf = np.array(scores)
    det = np.column_stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], det_conf))
    return det

if __name__ == '__main__':
    # --- Configuration ---
    VIDEO_INPUT = 'test_video.mp4'   # Put a low-light video in your folder
    VIDEO_OUTPUT = 'result_video.mp4'
    CONFIDENCE_THRESHOLD = 0.5       # Only draw boxes with > 50% confidence
    MY_SHRINK = 1

    # Load Model
    print("Loading model...")
    net = build_net('test', num_classes=2, model='dark')
    net.load_state_dict(torch.load('weights/dsfd.pth', map_location='cuda' if use_cuda else 'cpu'))
    net.eval()
    if use_cuda:
        net = net.cuda()

    # Open Video
    cap = cv2.VideoCapture(VIDEO_INPUT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    print(f"Processing Video: {width}x{height} at {fps} FPS")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # OpenCV reads in BGR, the original script expected RGB from PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            dets = detect_face(net, frame_rgb, MY_SHRINK)

        inference_time = time.time() - start_time
        current_fps = 1.0 / inference_time

        # Draw Bounding Boxes
        if len(dets) > 0:
            for i in range(dets.shape[0]):
                score = dets[i][4]
                if score > CONFIDENCE_THRESHOLD:
                    xmin, ymin, xmax, ymax = map(int, dets[i][0:4])
                    
                    # Draw Rectangle
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Draw Score Background & Text
                    label = f"Person: {score:.2f}"
                    cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 120, ymin), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw FPS on top corner
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1
        print(f"Processed Frame {frame_count} | FPS: {current_fps:.1f}", end='\r')

    cap.release()
    out.release()
    print("\nVideo processing complete! Saved to", VIDEO_OUTPUT)