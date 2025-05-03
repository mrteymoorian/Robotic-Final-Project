import cv2
import argparse
import numpy as np
from VideoStream import VideoStream
from Human_Detection import Human_Detection
from FeatureExtraction import SuperPointFrontend
from utils.datasets import letterbox

# Initialize the human detector and feature extractor
H_detector = Human_Detection()
fe = SuperPointFrontend(nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)

# Configuration variables
j = 0

def process_frame(frame):
    global j

    if j == 0:
        frame = letterbox(frame, 640, stride=64, auto=True)[0]
        H_detector.conf_box = [int(frame.shape[1] / 2) - 120, 0,
                               int(frame.shape[1] / 2) + 120, frame.shape[0]]

    j += 1

    # Configuration Loop
    frame_to_skip = 10

    if j <= 25:
        frame = letterbox(frame, 640, stride=64, auto=True)[0]
        frame_copy = frame.copy()
        H_detector.draw(frame_copy, H_detector.conf_box)
        cv2.imwrite(f'/home/human/output/conf_bbox_{j}.jpg', frame_copy)

        H_detector.detect(frame)
        best_box, best_mask = H_detector.configuration()
        if best_box is None or best_mask is None:
            return

        masked_img = H_detector.mask_bg(frame, best_box, best_mask)
        gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        pts, desc, _ = fe.run(gray_masked)

        indices = np.where(pts[2, :] >= 0.2)
        indices = indices[0]
        desc = desc[:, indices]

        H_detector.features = np.concatenate((H_detector.features, desc), axis=1)

        for i in range(pts.shape[1]):
            cv2.circle(masked_img, (int(pts[0][i]), int(pts[1][i])), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(f'/home/human/output/mask_{j}.jpg', masked_img)

    frame = letterbox(frame, 640, stride=64, auto=True)[0]
    H_detector.detect(frame)
    if H_detector.isdetected():
        bestmatches = 0
        best_box, best_mask = None, None
        for one_mask, bbox, cls, conf in H_detector.zipfile:
            if conf < H_detector.pred_conf or cls != H_detector.person_class_idx:
                continue
            masked_img = H_detector.mask_bg(frame, bbox, one_mask)
            gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            pts, desc, _ = fe.run(gray_masked)
            if desc is None or not desc.any():
                continue
            indices = np.where(pts[2, :] >= 0.2)
            indices = indices[0]
            desc = desc[:, indices]
            matches = fe.nn_match_two_way(H_detector.features, desc, 0.7)
            H_detector.draw(frame, bbox, one_mask, color=(255, 0, 0))
            if matches.shape[1] > bestmatches:
                bestmatches = matches.shape[1]
                best_box, best_mask = bbox, one_mask
        if bestmatches > 5:
            H_detector.draw(frame, best_box, best_mask, color=(0, 255, 0))
        print(bestmatches)
        cv2.imwrite(f'/home/human/output/tracking_{j}.jpg', frame)

def main():
    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument('--cam_idx', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--img_H', type=int, default=480,
                        help='Input image height (default: 480)')
    parser.add_argument('--img_W', type=int, default=640,
                        help='Input image width (default: 640)')

    args = parser.parse_args()

    # Initialize the video stream
    camera = VideoStream(camera_ID=args.cam_idx, img_width=args.img_W, img_height=args.img_H)
    if not camera.camera_isON():
        print("Error: Could not open webcam.")
        return

    while camera.camera_isON():
        frame = camera.next_frame()
        if frame is None:
            print("Error: Could not read frame.")
            break

        process_frame(frame)

        # Display the frame
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.camera_OFF()

    camera.camera_OFF()

if __name__ == '__main__':
    main()
