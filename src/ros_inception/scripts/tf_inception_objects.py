#!/usr/bin/env python 
import rospy
import tensorflow as tf
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow.models.image.imagenet import classify_image

class ObjectClassifier(object):
    

    def __init__(self):                
        self._BRIDGE = CvBridge()

        rospy.init_node('object_classifier')
        self._sub = rospy.Subscriber('raspi_image_raw', Image, self._classify_cb)
        self._string_pub = rospy.Publisher('object_detected', String, queue_size=1)
        self._image_pub = rospy.Publisher('image_detected', Image, queue_size=1)

        #set tf graph
        self._load_graph()
        
        #start session
        self._sess = tf.Session()
        
        #set parameters        
        self._score_threshold = rospy.get_param('~threshold', 0.05)
        self._k = rospy.get_param('~k',3)

    def _load_graph(self):
        classify_image.maybe_download_and_extract()
        classify_image.create_graph()

    def _show_image(self, image, string):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, string, (10,100), font, 1, (0,255,0),3)

        try:
            msg_img = self._BRIDGE.cv2_to_imgmsg(image, encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("cv_bridge exception: {}".format(e))

        self._image_pub.publish(msg_img)
    
    def _classify_cb(self, image):
        try:
            cv_img = self._BRIDGE.imgmsg_to_cv2(image)
        except CvBridgeError as e:
            rospy.logerr("cv_bridge exception: {}".format(e))

        img_data = cv2.imencode('.jpg', cv_img)[1].tostring()
        softmax = self._sess.graph.get_tensor_by_name('softmax:0')
        pred = self._sess.run(softmax, 
                            {'DecodeJpeg/contents:0': img_data})

        pred = np.squeeze(pred)    

        lookup = classify_image.NodeLookup()
        top_k = pred.argsort()[-self._k:][::-1]

        ss = []
        for id in top_k:
            ret_str = lookup.id_to_string(id)
            score = pred[id]
            if score > self._score_threshold:
                ss.append("[{:0.4f} - {}]\n".format(score, ret_str))   

        self._string_pub.publish(','.join(ss))
        self._show_image(cv_img, ss[0][:-1])      
            
    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()

    
if __name__=='__main__':
    clf = ObjectClassifier()
    clf.run()


