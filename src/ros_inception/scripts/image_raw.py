#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError


class Camera(object):

    
    def __init__(self, port):
        self._BRIDGE = CvBridge()
        self._CAMERA_PORT = port
        self._CAMERA = cv2.VideoCapture(self._CAMERA_PORT)
        self._RAMP = 20

        rospy.init_node('camera_feed')
        self._image_pub = rospy.Publisher('image_raw', Image, queue_size=1)
        
    def _ramp_up(self):
        temp = [self._CAMERA.read() for _ in xrange(self._RAMP)]

    def run(self):
        rate = rospy.Rate(10)
        self._ramp_up()

        try:
            while not rospy.is_shutdown():
                
                test, cv_image = self._CAMERA.read()

                if test:
                    image_msg = self._BRIDGE.cv2_to_imgmsg(cv_image, encoding="bgr8")
                
                    self._image_pub.publish(image_msg)
                    rate.sleep()

        except rospy.ROSInterruptException:
            pass    


if __name__=='__main__':
    port = 0
    node = Camera(port)
    node.run()

