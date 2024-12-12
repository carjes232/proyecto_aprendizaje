#!/home/pi/venv_freenect/bin/python
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
import time
import subprocess
import freenect
import cv2
import frame_convert
import numpy as np
import logging
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import signal 

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize YOLO model (ensure you use the correct model path and define the task explicitly)
model = YOLO("/home/pi/proyecto_aprendizaje/Raspberry/models/yolo/yolov8n_ncnn_model", task="detect")

class KinectYoloNode(Node):
    def __init__(self):
        super().__init__('kinect_yolo_node')
        # Create publishers for the Kinect images (RGB frames and annotated frames)
        self.publisher_image = self.create_publisher(Image, 'kinect_image', 10)
        self.publisher_result = self.create_publisher(Image, 'result_image', 10)
        
        # Timer setup (1-second interval)
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Logging
        self.get_logger().info('Kinect YOLO Node has been started')
        
        # Retry mechanism parameters
        self.retry_delay = 2  # seconds
        self.max_retries = 5
        self.retry_count = 0

    def timer_callback(self):
        try:
            # Attempt to capture a frame from the Kinect (RGB)
            result = freenect.sync_get_video()
            if result is None:
                self.get_logger().warn('No RGB frame captured from Kinect.')
                raise ValueError("freenect.sync_get_video() returned None")
            
            rgb, _ = result  # Unpack the result
            if rgb is None:
                self.get_logger().warn('RGB frame is None.')
                return

            # Reset retry count upon successful capture
            self.retry_count = 0

            # Convert the RGB frame to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
            self.publisher_image.publish(ros_image)
            logger.info('Antes de entrar al modelo')
            # Perform YOLO inference on the captured frame
            results = model(rgb)
            annotated_frame = results[0].plot()
            
            # Save the annotated frame (optional)
            cv2.imwrite('/home/pi/output/funciona.png', annotated_frame)
            
            # Convert the annotated frame to a ROS Image message
            ros_image2 = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.publisher_result.publish(ros_image2)
            
            # Log successful processing
            self.get_logger().info('Frame processed and published successfully.')

        except ValueError as ve:
            self.get_logger().error(f"ValueError in timer_callback: {ve}")
            self.handle_usb_busy()
        except Exception as e:
            self.get_logger().error(f"Unexpected error in timer_callback: {e}")
            self.handle_usb_busy()

    def kill_stale_processes(self):
        try:
            # Find any processes holding /dev/bus/usb devices
            output = subprocess.check_output("lsof | grep /dev/bus/usb", shell=True).decode('utf-8')
            
            # Parse each line for a PID
            for line in output.split('\n'):
                if not line.strip():
                    continue
                parts = line.split()
                # Typically 'parts[1]' is the PID
                if len(parts) > 1 and parts[1].isdigit():
                    pid_int = int(parts[1])
                    # Skip this process if it's the current one
                    if pid_int == os.getpid():
                        continue
                    
                    self.get_logger().info(f"Sending SIGTERM to stale process with PID {pid_int}")
                    try:
                        os.kill(pid_int, signal.SIGTERM)
                        time.sleep(1)
                        if self.is_process_running(pid_int):
                            self.get_logger().info(f"Process {pid_int} did not terminate, sending SIGKILL")
                            os.kill(pid_int, signal.SIGKILL)
                    except ProcessLookupError:
                        self.get_logger().warn(f"Process {pid_int} no longer exists.")
                    except Exception as kill_err:
                        self.get_logger().error(f"Failed to kill process {pid_int}: {kill_err}")
        except subprocess.CalledProcessError:
            # Means lsof or grep returned no matches
            self.get_logger().info("No stale processes found holding /dev/bus/usb")
        except Exception as e:
            self.get_logger().error(f"Error identifying/killing stale processes: {e}")


    def is_process_running(self, pid):
        """
        Checks if a process with the given PID is still running.
        """
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def reset_usb_device(self, vid, pid):
        try:
            # Find the device path using lsusb
            lsusb_output = subprocess.check_output(['lsusb']).decode()
            for line in lsusb_output.split('\n'):
                if f"{vid}:{pid}" in line:
                    parts = line.split()
                    bus = parts[1]
                    device = parts[3].rstrip(':')
                    device_path = f"/dev/bus/usb/{bus}/{device}"
                    self.get_logger().info(f"Resetting USB device at {device_path}")
                    # Reset the USB device using usbreset tool
                    # Ensure usbreset is installed and available in PATH
                    subprocess.check_call(['usbreset', device_path])
                    return True
            self.get_logger().warn("USB device not found for reset.")
            return False
        except subprocess.CalledProcessError as cpe:
            self.get_logger().error(f"usbreset failed: {cpe}")
            return False
        except Exception as e:
            self.get_logger().error(f"Failed to reset USB device: {e}")
            return False

    def handle_usb_busy(self):
        """Handle USB busy error by killing stale processes, resetting the USB device and then retrying."""
        VID = "045e"  # Kinect Vendor ID
        PID = "02ae"  # Kinect Product ID
        
        # Try to kill any stale processes first
        self.kill_stale_processes()

        if self.retry_count >= self.max_retries:
            self.get_logger().error("Max retries reached. Shutting down node.")
            self.destroy_node()
            rclpy.shutdown()
            return

        if self.reset_usb_device(VID, PID):
            self.get_logger().info("USB device reset successfully.")
        else:
            self.get_logger().warn("USB device reset failed or not necessary.")

        self.retry_count += 1
        self.get_logger().info(f"Retrying to access Kinect in {self.retry_delay} seconds... (Attempt {self.retry_count}/{self.max_retries})")
        time.sleep(self.retry_delay)

def main(args=None):
    rclpy.init(args=args)
    kinect_yolo_node = KinectYoloNode()

    try:
        # Spin the node to keep it running
        rclpy.spin(kinect_yolo_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown the node when the user interrupts (Ctrl+C)
        kinect_yolo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
