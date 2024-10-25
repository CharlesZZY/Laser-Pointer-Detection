# tests/test_detection.py

import unittest
from typing import Optional

import cv2
import os
from src.laser_detection import LaserDetector
from src.utils import draw_laser_point


class TestLaserDetection(unittest.TestCase):
    def setUp(self) -> None:
        """
        Initializes the test environment, including creating a laser detection instance
        and defining the test image path.
        """
        # Create an instance of the laser detector
        self.detector: LaserDetector = LaserDetector()

        # Use relative path to locate the test images folder
        self.test_images_path: str = os.path.join(os.path.dirname(__file__), '..', 'images')
        self.test_image_path: str = os.path.join(self.test_images_path, 'test.jpg')

        # Ensure the output folder exists to save detection results
        self.output_path: str = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(self.output_path, exist_ok=True)

    def test_detect_laser_and_save_output(self) -> None:
        """
        Tests the laser point detection function. If a laser point is detected,
        marks the position and distance on the image. Saves the processed image
        and masks to the output folder.
        """
        for img_path in [self.test_image_path]:
            with self.subTest(img_path=img_path):
                # Load the test image and confirm it loads successfully
                image: cv2.Mat = cv2.imread(img_path)
                self.assertIsNotNone(image, f"Failed to load image: {img_path}")

                # Perform laser point detection
                laser_data: Optional[tuple[int, int, float, cv2.Mat]] = self.detector.detect_laser(image)

                # If a laser point is detected, mark the position and distance
                if laser_data:
                    x, y, radius, largest_contour = laser_data
                    distance: float = self.detector.estimate_distance(radius)
                    draw_laser_point(image, x, y, distance)

                # Save the processed result image
                output_filename: str = f"output_{os.path.basename(img_path)}"
                output_file_path: str = os.path.join(self.output_path, output_filename)
                cv2.imwrite(output_file_path, image)
                print(f"Processed image saved to: {output_file_path}")

                # Retrieve and save the generated masks
                red_mask, thresholded_gray, combined_mask = self.detector.get_masks()

                if red_mask is not None:
                    cv2.imwrite(os.path.join(self.output_path, f"red_mask_{os.path.basename(img_path)}"), red_mask)
                if thresholded_gray is not None:
                    cv2.imwrite(os.path.join(self.output_path, f"thresholded_gray_{os.path.basename(img_path)}"),
                                thresholded_gray)
                if combined_mask is not None:
                    cv2.imwrite(os.path.join(self.output_path, f"combined_mask_{os.path.basename(img_path)}"),
                                combined_mask)

                print(f"Masks saved to output folder for image: {img_path}")

                cv2.imshow("Laser Pointer Detection", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def test_estimate_distance(self) -> None:
        """
        Tests the distance estimation function, confirming the estimated distance is within a reasonable range.
        """
        # Test laser point radius values
        test_radii: list[int] = [5, 10, 15, 20]
        for radius in test_radii:
            with self.subTest(radius=radius):
                distance: float = self.detector.estimate_distance(radius)
                # Verify that the estimated distance is reasonable
                self.assertGreater(distance, 0, "Estimated distance should be positive")
                self.assertLessEqual(distance, self.detector.distance_threshold,
                                     "Estimated distance should be within the threshold")


if __name__ == "__main__":
    unittest.main()
