import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from Human_Detection import *
from FeatureExtraction import *
from utils.datasets import letterbox
import math
from controller import PID


def image_to_histogram(image, bins=32):
    # Convert image to histogram with specified bins for each color channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = np.histogramdd(image.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])[0]
    # Normalize the histogram
    histogram = histogram.ravel()
    histogram /= histogram.sum()
    return histogram


class HumanTracker:
    def __init__(self):
        self.detector = Human_Detection()
        self.fe = SuperPointFrontend(nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        self.angular_vel = 0
        self.linear_PID = PID(.2, 0.0001, 0)
        self.angular_PID = PID(.015, 0, 0)
        self.x_thresh = 10
        self.y_thresh = 5
        self.desired_distance = 80
        self.FOV_x = 87 #deg
        self.FOV_y = 58 #deg
        self.rgb_histo = []

    def depth_callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "16UC1")
        self.depth_image = image
        self.depth_time = image_msg.header.stamp.secs

    def callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8") 

        if not hasattr(self, 'frame_count'):
            self.initialize(image)

        if self.frame_count <= 40:
            self.configure(image)
        else:
            self.track(image)

        self.frame_count += 1

    def initialize(self, image):
        self.frame_count = 0
        self.detector.conf_box = [int(image.shape[1]/2)-120, 0, int(image.shape[1]/2)+120, image.shape[0]]
    
    def configure(self, image):
        frame = letterbox(image, 640, stride=64, auto=True)[0]
        self.detector.detect(frame)
        best_box, best_mask = self.detector.configuration()
        if best_box is None or best_mask is None:
            return
        masked_img = self.detector.mask_bg(frame, best_box, best_mask)
        gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        pts, desc, _ = self.fe.run(gray_masked) 
        indices = np.where(pts[2, :] >= 0.2)[0]
        desc = desc[:, indices]
        self.detector.features = np.concatenate((self.detector.features, desc), axis=1)
        for i in range(pts.shape[1]):
            cv2.circle(masked_img, (int(pts[0][i]), int(pts[1][i])), radius=2, color=(0, 0, 255), thickness=-1)

        histo = image_to_histogram(masked_img)
        histo /= np.linalg.norm(histo)
        self.rgb_histo.append(histo)

    def track(self, image):

        frame = letterbox(image, 640, stride=64, auto=True)[0]
        self.detector.detect(frame)
        best_matches = 0
        best_box, best_mask = None, None
        best_histo = 10

        for one_mask, bbox, cls, conf in self.detector.zipfile:
            if conf < self.detector.pred_conf or cls != self.detector.person_class_idx:
                continue
            masked_img = self.detector.mask_bg(frame, bbox, one_mask)
            gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            pts, desc, _ = self.fe.run(gray_masked)
            if desc is None or not desc.any():
                continue
            indices = np.where(pts[2, :] >= 0.2)[0]
            desc = desc[:, indices]
            matches = self.fe.nn_match_two_way(self.detector.features, desc, 0.7)
            self.detector.draw(frame, bbox, one_mask, color=(255, 0, 0))
            histo = image_to_histogram(masked_img)
            histo /= np.linalg.norm(histo)
            minn = 10.0
            for i in range(len(self.rgb_histo)):
                minn = min(minn, np.linalg.norm(histo - self.rgb_histo[i]))
            if(minn < best_histo) and matches.shape[1] > best_matches:
                best_matches = matches.shape[1]
                best_box, best_mask = bbox, one_mask
                best_histo = minn
            
        print(best_matches, best_histo)

        if best_matches > 5 and best_histo < 0.2:
            self.detector.draw(frame, best_box, best_mask, color=(0, 255, 0))
            cx = (best_box[0] + best_box[2]) / 2
            cy = (best_box[1] + best_box[3]) / 2
            # center distance
            # image width in mm

            depth = self.depth_image[int(cy)][int(cx)]
            distance_x = (depth/10)*math.cos(math.radians(50))
            distance_error_x = distance_x - self.desired_distance
            distance_error_y_pxl = cx - 320
            
            distance_error_y = distance_error_y_pxl * (self.FOV_y/640)
            
            #print(distance_x, distance_error_y)
            linear_speed = self.linear_PID.update(distance_error_x/21)

            angular_speed = self.angular_PID.update(distance_error_y/(-math.pi/4))
            twist_msg = Twist()
            twist_msg.angular.z = angular_speed
            twist_msg.linear.x = linear_speed
            if abs(distance_error_x) < self.x_thresh and abs(distance_error_y) < self.y_thresh:
                return

            self.pub.publish(twist_msg)
            self.rate.sleep()
            

        
        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)


def main():
    rospy.init_node('human_tracker', anonymous=True)
    tracker = HumanTracker()
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, tracker.depth_callback)
    rospy.Subscriber('/camera/color/image_raw', Image, tracker.callback)
    rospy.spin()

if __name__ == '__main__':
    main()