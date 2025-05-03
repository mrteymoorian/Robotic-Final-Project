# import cv2
# import argparse
# import time
# import numpy as np
# from VideoStream import *
# from Human_Detection import *
# from FeatureExtraction import *
# from utils.datasets import letterbox
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# j = 0
# H_detector = Human_Detection()
# fe = SuperPointFrontend(nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
# bridge = CvBridge()

# def callback(image):
#     global j
#     # print(type(image.data), len(image.data))
#     image = bridge.imgmsg_to_cv2(image, "bgr8")
#     frame = image
    
#     if j == 0:
#         frame = letterbox(frame, 640, stride=64, auto=True)[0]
#         H_detector.conf_box = [int(frame.shape[1] / 2) - 120, 0,
#                                int(frame.shape[1] / 2) + 120, frame.shape[0]]

#     j += 1
#     frame_to_skip = 10

#     if j <= 25:
#         frame = letterbox(frame, 640, stride=64, auto=True)[0]
#         frame_copy = frame.copy()
#         H_detector.draw(frame_copy, H_detector.conf_box)
#         # Display the frame with the bounding box
#         cv2.imshow("Configuration BBox", frame_copy)
#         cv2.waitKey(1)
        
#         H_detector.detect(frame)
#         best_box, best_mask = H_detector.configuration()
#         if best_box is None or best_mask is None:
#             return
        
#         masked_img = H_detector.mask_bg(frame, best_box, best_mask)
#         gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
#         pts, desc, _ = fe.run(gray_masked) 
        
#         indices = np.where(pts[2, :] >= 0.2)
#         indices = indices[0]
#         desc = desc[:, indices]

#         H_detector.features = np.concatenate((H_detector.features, desc), axis=1)
            
#         for i in range(pts.shape[1]):
#             cv2.circle(masked_img, (int(pts[0][i]), int(pts[1][i])), radius=2, color=(0, 0, 255), thickness=-1)
#         # Display the masked image with features
#         cv2.imshow("Masked Image", masked_img)
#         cv2.waitKey(1)

#     print("Recorded {} features of the person".format(H_detector.features.shape[1]))

#     frame = image
#     frame = letterbox(frame, 640, stride=64, auto=True)[0]
    
#     H_detector.detect(frame)
#     if H_detector.isdetected():
#         bestmatches = 0
#         best_box, best_mask = None, None
#         for one_mask, bbox, cls, conf in H_detector.zipfile:
#             if conf < H_detector.pred_conf or cls != H_detector.person_class_idx:
#                 continue
#             masked_img = H_detector.mask_bg(frame, bbox, one_mask)
#             gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
#             pts, desc, _ = fe.run(gray_masked)
#             if desc is None or not desc.any():
#                 continue   
#             indices = np.where(pts[2, :] >= 0.2)
#             indices = indices[0]
#             desc = desc[:, indices]
#             matches = fe.nn_match_two_way(H_detector.features, desc, 0.7)
#             H_detector.draw(frame, bbox, one_mask, color=(255, 0, 0))
#             if matches.shape[1] > bestmatches:
#                 bestmatches = matches.shape[1]
#                 best_box, best_mask = bbox, one_mask
#         if bestmatches > 5:
#             H_detector.draw(frame, best_box, best_mask, color=(0, 255, 0))
#         print(bestmatches)
#         # Display the tracking frame
#         cv2.imshow("Tracking Frame", frame)
#         cv2.waitKey(1)

# # Ensure to properly initialize the ROS node and subscribe to the topic
# rospy.init_node('human_detection_node', anonymous=True)
# rospy.Subscriber("/camera/image_raw", Image, callback)

# # Keep the node running
# rospy.spin()

# # Cleanup: Destroy all windows when done
# cv2.destroyAllWindows()

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Initialize the CvBridge
bridge = CvBridge()

def callback(image):
    # Convert ROS Image message to OpenCV image
    frame = bridge.imgmsg_to_cv2(image, "bgr")

    print(frame[240,320]/10)
    
    # Display the image
    # cv2.imshow("ROS Image Stream", frame)
    # cv2.waitKey(1)  # Necessary to update the display window

def main():
    # Initialize the ROS node
    rospy.init_node('image_streamer', anonymous=True)
    
    # Subscribe to the image topic
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, callback)

    # Keep the node running
    rospy.spin()
    
    # Cleanup: Destroy all windows when done
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
