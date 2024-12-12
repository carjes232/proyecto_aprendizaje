import os
import sys

# Path to virtual environment
venv_path = "/home/pi/venv_freenect"

# Add virtual environment's site-packages to sys.path
venv_site_packages = os.path.join(venv_path, "lib", "python3.10", "site-packages")
sys.path.insert(0, venv_site_packages)

# Add virtual environment's lib to LD_LIBRARY_PATH
lib_path = os.path.join(venv_path, "lib")
os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Add current directory to sys.path to find frame_convert
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now import the rest of the modules
import freenect
import cv2
import frame_convert
import time
import numpy as np
import logging
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize YOLO model (ensure you use the correct model path)
model = YOLO("/home/pi/proyecto_aprendizaje/Raspberry/models/yolo/yolov8n_ncnn_model")

class KinectYoloNode(Node):
    def __init__(self):
        super().__init__('kinect_yolo_node')
        # Create a publisher for the Kinect images (RGB frames)
        self.publisher_ = self.create_publisher(Image, 'kinect_image', 1)
        self.publisher_ = self.create_publisher(Image, 'result_image', 1)
        self.timer = self.create_timer(1, self.timer_callback)  
        self.bridge = CvBridge()  # Convert OpenCV images to ROS messages
        self.get_logger().info('Kinect YOLO Node has been started')

    def timer_callback(self):
        try:
            # Capture a frame from the Kinect (RGB)
            rgb, _ = freenect.sync_get_video()  # RGB frame from Kinect
            if rgb is None:
                self.get_logger().warn('No RGB frame captured from Kinect.')
                return

            # Convert the RGB frame to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
            self.publisher_.publish(ros_image)

            # Perform YOLO inference on the captured frame
            results = model(rgb)
            annotated_frame = results[0].plot()
            # Save the annotated frame
            cv2.imwrite('/home/pi/output/funciona.png', annotated_frame)
            ros_image2 = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
            self.publisher_.publish(ros_image2)
            # Log the results (you can process them further if needed)
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    kinect_yolo_node = KinectYoloNode()

    try:
        # Spin the node to keep it running
        rclpy.spin(kinect_yolo_node)
    except KeyboardInterrupt:
        pass

    # Shutdown the node when the user interrupts (Ctrl+C)
    kinect_yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
