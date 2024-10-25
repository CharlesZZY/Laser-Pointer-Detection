# src/laser_detection.py
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from src.config import CONFIG


class LaserDetector:
    def __init__(self, frame_width=640, frame_height=480):
        """
        Initializes the laser detector class, setting parameters like color threshold,
        distance threshold, and image frame dimensions.

        Parameters:
            frame_width (int): Width of the image frame, default is 640 pixels.
            frame_height (int): Height of the image frame, default is 480 pixels.
        """
        # Initialize properties to store masks
        self.combined_mask: np.ndarray | None = None
        self.red_mask: np.ndarray | None = None
        self.thresholded_gray: np.ndarray | None = None

        # Load HSV threshold values for red laser from the configuration file
        self.color_lower1: np.ndarray = np.array(CONFIG["laser_color_range"]["lower1"], dtype=np.uint8)
        self.color_upper1: np.ndarray = np.array(CONFIG["laser_color_range"]["upper1"], dtype=np.uint8)
        self.color_lower2: np.ndarray = np.array(CONFIG["laser_color_range"]["lower2"], dtype=np.uint8)
        self.color_upper2: np.ndarray = np.array(CONFIG["laser_color_range"]["upper2"], dtype=np.uint8)

        # Set the distance threshold for the laser point and the minimum contour area threshold
        self.distance_threshold: float = CONFIG["distance_threshold"]
        self.area_threshold: int = 5  # Area threshold to filter out small noise
        self.intensity_threshold: int = 200  # Initial grayscale threshold value for dynamic adjustment

        # Use tkinter to get screen resolution for window layout
        root = tk.Tk()
        self.screen_width: int = root.winfo_screenwidth()
        self.screen_height: int = root.winfo_screenheight()
        root.destroy()

        # Set the frame width and height
        self.frame_width: int = frame_width
        self.frame_height: int = frame_height

        # Initialize window position and size
        self._initialize_windows()
        self.first_display: bool = True

    def _initialize_windows(self):
        """
        Initializes display windows, ensuring proportional scaling across different screen resolutions.
        """
        # Calculate the aspect ratio of the window for proportional scaling
        aspect_ratio: float = self.frame_width / self.frame_height
        max_width: int = self.screen_width // 2
        max_height: int = self.screen_height // 2

        # Adjust window size according to screen size while maintaining the image aspect ratio
        if max_width / aspect_ratio <= max_height:
            scaled_width: int = max_width
            scaled_height: int = int(max_width / aspect_ratio)
        else:
            scaled_width = int(max_height * aspect_ratio)
            scaled_height = max_height

        # Create three windows to display different masks
        cv2.namedWindow("Red Mask", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Thresholded Gray Mask", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Combined Mask", cv2.WINDOW_AUTOSIZE)

        # Set window size and position to display in three corners of the screen without overlapping
        cv2.resizeWindow("Red Mask", scaled_width, scaled_height)
        cv2.resizeWindow("Thresholded Gray Mask", scaled_width, scaled_height)
        cv2.resizeWindow("Combined Mask", scaled_width, scaled_height)
        cv2.moveWindow("Red Mask", self.screen_width - scaled_width, 0)  # Top-right corner
        cv2.moveWindow("Thresholded Gray Mask", 0, self.screen_height - scaled_height)  # Bottom-left corner
        cv2.moveWindow("Combined Mask", self.screen_width - scaled_width,
                       self.screen_height - scaled_height)  # Bottom-right corner

    def detect_laser(self, frame: np.ndarray) -> Optional[tuple[int, int, float, np.ndarray]]:
        """
        Detects the laser point's position and size by generating grayscale, red, and combined masks
        for precise localization of the laser point.

        Parameters:
            frame (numpy.ndarray): Input image frame.

        Returns:
            tuple: If a laser point is detected, returns its coordinates (x, y), radius, and largest contour;
                   otherwise, returns None.
        """
        try:
            # Convert the image to grayscale and compute a dynamic threshold
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.intensity_threshold: int = self._adaptive_intensity_threshold(gray_frame)
            _, self.thresholded_gray = cv2.threshold(gray_frame, self.intensity_threshold, 255, cv2.THRESH_BINARY)

            # Convert the image to HSV color space to generate a red mask
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv_frame, self.color_lower1, self.color_upper1)
            mask2 = cv2.inRange(hsv_frame, self.color_lower2, self.color_upper2)
            self.red_mask = cv2.bitwise_or(mask1, mask2)  # Combine the red mask ranges

            # Create a combined mask by combining the red and grayscale masks to filter out non-laser noise
            self.combined_mask = cv2.bitwise_and(self.thresholded_gray, self.red_mask)
            self.combined_mask = self._apply_multilevel_morphology(self.combined_mask)  # Apply morphological operations

            # Display generated masks
            cv2.imshow("Red Mask", self.red_mask)
            cv2.imshow("Thresholded Gray Mask", self.thresholded_gray)
            cv2.imshow("Combined Mask", self.combined_mask)

            # Detect contours in the combined mask to find the largest, most prominent laser point
            contours, _ = cv2.findContours(self.combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Select the largest contour as the laser point
                best_contour = max(contours, key=lambda cnt: (cv2.contourArea(cnt), cv2.minEnclosingCircle(cnt)[1]))
                area = cv2.contourArea(best_contour)
                if area < self.area_threshold:
                    return None  # Treat as noise if the contour area is too small

                # Calculate the laser point's center position and radius
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                return int(x), int(y), radius, best_contour
        except Exception as e:
            print(f"Error in detect_laser: {e}")
        return None

    def estimate_distance(self, radius: float) -> float:
        """
        Estimates the distance to the camera based on the detected laser point radius.

        Parameters:
            radius (float): Radius of the laser point (in pixels).

        Returns:
            float: Estimated distance from the laser point to the camera (in cm).
        """
        try:
            if radius > 0:
                # Calculate distance based on inverse relation, avoiding division by zero
                distance: float = 1500 / (radius + 0.1)
            else:
                distance = self.distance_threshold  # Use default threshold if no valid radius is detected
            return min(distance, self.distance_threshold)  # Limit the distance to the threshold
        except Exception as e:
            print(f"Error in estimate_distance: {e}")
            return self.distance_threshold

    def _apply_multilevel_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies multi-level morphological operations on the mask to reduce noise and highlight the laser point.

        Parameters:
            mask (numpy.ndarray): Input binary mask image.

        Returns:
            numpy.ndarray: Processed binary mask.
        """
        try:
            # Dilate the mask to remove isolated small noise
            dilate_kernel1: np.ndarray = np.ones((3, 3), np.uint8)
            dilated_mask: np.ndarray = cv2.dilate(mask, dilate_kernel1, iterations=2)

            # Use a smaller erosion kernel to further denoise
            erode_kernel1: np.ndarray = np.ones((2, 2), np.uint8)
            eroded_mask: np.ndarray = cv2.erode(dilated_mask, erode_kernel1, iterations=1)

            # Use a larger dilation kernel to enhance laser point clarity
            dilate_kernel2: np.ndarray = np.ones((4, 4), np.uint8)
            final_mask: np.ndarray = cv2.dilate(eroded_mask, dilate_kernel2, iterations=1)
            return final_mask
        except Exception as e:
            print(f"Error in _apply_multilevel_morphology: {e}")
            return mask  # Return the original mask if morphological operation fails

    def _adaptive_intensity_threshold(self, gray_frame: np.ndarray) -> int:
        """
        Dynamically adjusts the grayscale threshold to adapt to changes in ambient lighting,
        enabling more accurate laser point detection.

        Parameters:
            gray_frame (numpy.ndarray): Input grayscale image.

        Returns:
            int: Dynamically adjusted grayscale threshold.
        """
        mean_intensity = np.mean(gray_frame)
        # Calculate the threshold based on average brightness, constrained between 180 and 255
        adjusted_threshold = max(180, min(255, int(mean_intensity * 1.5)))
        return adjusted_threshold

    def get_masks(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns the currently generated masks for debugging and testing.

        Returns:
            tuple: A tuple containing the red mask, thresholded gray mask, and combined mask.
        """
        return self.red_mask, self.thresholded_gray, self.combined_mask
