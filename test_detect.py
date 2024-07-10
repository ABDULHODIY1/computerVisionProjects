import tkinter as tk
from tkinter import ttk
import torch
import cv2
import datetime
import requests
import serial
import numpy as np
from snap7 import util, client
from snap7.util import *
import snap7

class HumanDetector:
    """
    Class for human detection
    """
    def __init__(self, camera_index=0, usb_ports=None):
        """
        Initialize the detector
        """
        self.url = "192.168.192.91"
        self.port = "8080"
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.cap = cv2.VideoCapture(camera_index)
        self.out = cv2.VideoWriter(
            'output.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            15.,
            (640, 480))
        self.people_count = 0
        self.usb_ports = usb_ports if usb_ports else []

        # Initialize PLC
        self.plc = client.Client()
        self.plc.set_connection_type(3)
        self.plc.connect("192.168.1.11", 0, 1)
        self.plc_state = self.plc.get_connected()
        print(f"PLC connected: {self.plc_state}")

    def mWriteBool(self, byte, bit, value):
        """
        Write a boolean value to the PLC
        """
        data = self.plc.read_area(snap7.types.Areas.MK, 0, byte, 1)
        set_bool(data, 0, bit, value)
        self.plc.write_area(snap7.types.Areas.MK, 0, byte, data)

    def detect(self):
        """
        Detect people
        :return:
        """
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Get current hour
            now = datetime.datetime.now()
            hour = now.hour

            # Resizing for faster detection
            frame = cv2.resize(frame, (640, 480))

            # Model yordamida aniqlash
            results = self.model(frame)

            # Annotatsiyalarni olish
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

            # Aniqlangan odamlar sonini sanash
            self.people_count = 0
            for i in range(len(labels)):
                if labels[i] == 0:  # 0 - 'person' klassi
                    self.people_count += 1

            # Annotatsiyalarni chizish
            results.render()

            # Check and send signal
            if self.signal(hour, self.people_count):
                print(f"Signal sent at {now.strftime('%Y-%m-%d %H:%M:%S')} with {self.people_count} people detected.")
                self.people_count = 0  # Reset the count after sending the signal

            # Display people count on the frame
            cv2.putText(frame, f'People Count: {self.people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Write the output video
            self.out.write(frame.astype('uint8'))

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def send_signal(self, value, method="serial", port_name='/dev/ttyUSB0'):
        """
        Send signal via port or PLC
        :param value: Value to send
        :param method: Method of signal sending
        :param port_name: Serial Port Name
        :return: None
        """
        if method == "network":
            try:
                response = requests.post(f"http://{self.url}:{self.port}/signal", json={"signal": value})
                if response.status_code == 200:
                    print("Signal sent via network.")
                else:
                    print(f"Failed to send signal via network: {response.status_code}")
            except Exception as e:
                print(f"Network error: {e}")

        elif method == "serial":
            try:
                ser = serial.Serial(port_name, 9600, timeout=1)
                ser.write(f"{value}\n".encode())
                ser.close()
                print("Signal sent via serial port.")
            except Exception as e:
                print(f"Serial error: {e}")

        elif method == "plc":
            try:
                self.mWriteBool(5, 0, value)
                print("Signal sent via PLC.")
            except Exception as e:
                print(f"PLC error: {e}")

    def signal(self, hour, count):
        """
        Analyze number of people and time
        :param hour: Hour
        :param count: Number of people
        :return: Boolean
        """
        if 6 <= hour < 17 and count >= 8:
            self.send_signal(True, method="plc")
            print(True)
            return True
        elif 17 <= hour < 18 and count >= 4:
            self.send_signal(True, method="plc")
            print(True)
            return True
        elif hour >= 18 and count >= 1:
            self.send_signal(True, method="plc")
            print(True)
            return True
        return False

def detect_cameras():
    """
    Detect connected cameras
    """
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def detect_usb_ports():
    """
    Detect USB ports
    """
    ports = []
    for i in range(256):
        try:
            s = serial.Serial(f'COM{i}')
            s.close()
            ports.append(f'COM{i}')
        except (OSError, serial.SerialException):
            pass
    return ports

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detector Settings")

        # Detect cameras and USB ports
        self.cameras = detect_cameras()
        self.usb_ports = detect_usb_ports()

        # Camera selection
        self.camera_label = tk.Label(root, text="Select Camera:")
        self.camera_label.pack()
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(root, textvariable=self.camera_var)
        self.camera_dropdown['values'] = self.cameras + ['Default Camera']
        if self.cameras:
            self.camera_dropdown.current(0)
        else:
            self.camera_dropdown.set('Default Camera')
        self.camera_dropdown.pack()

        # USB port selection
        self.usb_labels = []
        self.usb_vars = []
        self.usb_dropdowns = []
        for i in range(1, 5):
            label = tk.Label(root, text=f"Select USB Port {i}:")
            label.pack()
            self.usb_labels.append(label)
            var = tk.StringVar()
            self.usb_vars.append(var)
            dropdown = ttk.Combobox(root, textvariable=var)
            dropdown['values'] = self.usb_ports
            if self.usb_ports:
                dropdown.current(0)
            dropdown.pack()
            self.usb_dropdowns.append(dropdown)

        # Start button
        self.start_button = tk.Button(root, text="Start Detector", command=self.start_detector)
        self.start_button.pack()

    def start_detector(self):
        camera_index = self.camera_var.get()
        if camera_index == 'Default Camera':
            camera_index = 0
        else:
            camera_index = int(camera_index)
        usb_ports = [var.get() for var in self.usb_vars]
        self.detector = HumanDetector(camera_index=camera_index, usb_ports=usb_ports)
        self.detector.detect()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
