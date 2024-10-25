# src/main.py
from typing import Optional

import cv2
from laser_detection import LaserDetector
from utils import draw_laser_point
import tkinter as tk


def main():
    """
    Main function to start the laser point detection application. This application
    detects the laser point position in real-time through the camera and displays
    the detected laser point's distance and coordinates on the screen.
    """

    # Create an instance of the laser detector
    detector = LaserDetector()

    # Use tkinter to get the screen resolution for window adjustment
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Set the default frame width and height to match the default values in LaserDetector
    frame_width, frame_height = 640, 480
    aspect_ratio = frame_width / frame_height
    max_width = screen_width // 2
    max_height = screen_height // 2

    # Scale down the window size proportionally to fit half the screen size
    if max_width / aspect_ratio <= max_height:
        scaled_width = max_width
        scaled_height = int(max_width / aspect_ratio)
    else:
        scaled_width = int(max_height * aspect_ratio)
        scaled_height = max_height

    # Start the camera (0 indicates the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a display window for showing the real-time detection output
    cv2.namedWindow("Laser Pointer Detection", cv2.WINDOW_AUTOSIZE)

    # Flag to adjust window position and size on the first display
    first_display: bool = True

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform laser point detection, returning the laser point's location information
        laser_data: Optional[tuple[int, int, float, cv2.Mat]] = detector.detect_laser(frame)

        # If a laser point is detected, draw its position and distance information on the frame
        if laser_data:
            x, y, radius, contour = laser_data
            distance = detector.estimate_distance(radius)
            draw_laser_point(frame, x, y, distance, contour)

        # Display the real-time detection output
        cv2.imshow("Laser Pointer Detection", frame)

        # On the first display, set window size and position
        if first_display:
            cv2.resizeWindow("Laser Pointer Detection", scaled_width, scaled_height)
            cv2.moveWindow("Laser Pointer Detection", 0, 0)  # Move to the top left corner
            first_display = False

        # Exit the program by pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera resources and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
