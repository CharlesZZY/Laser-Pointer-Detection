# src/utils.py

import cv2


def draw_laser_point(frame, x, y, distance, contour=None):
    """
    Marks the detected laser point on the image, displaying position, distance, and contour information.

    :param frame: The image frame to draw on
    :param x: x-coordinate of the laser point
    :param y: y-coordinate of the laser point
    :param distance: Distance from the laser point to the camera (in cm)
    :param contour: Contour of the laser point (used for drawing the contour)
    """
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Draw the laser point contour in green

    # Draw a small red dot at the laser point position
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    # Display distance information
    cv2.putText(frame, f"Distance: {distance:.2f} cm", (int(x) + 15, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Display coordinates of the laser point
    cv2.putText(frame, f"Position: ({int(x)}, {int(y)})", (int(x) + 15, int(y) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
