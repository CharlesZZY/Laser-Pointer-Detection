# Configuration settings for laser detection application
CONFIG = {
    "laser_color_range": {
        # HSV color range for detecting red laser light
        # The first range (lower1 to upper1) targets lower red hues
        "lower1": (0, 100, 160),  # Lower bound for red hue range 1
        "upper1": (10, 255, 255),  # Upper bound for red hue range 1

        # The second range (lower2 to upper2) targets higher red hues
        "lower2": (160, 100, 160),  # Lower bound for red hue range 2
        "upper2": (180, 255, 255)  # Upper bound for red hue range 2
    },
    "distance_threshold": 200,  # Maximum distance threshold (in cm) for laser detection
}
