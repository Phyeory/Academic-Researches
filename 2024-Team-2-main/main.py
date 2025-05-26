from exif import Image
from datetime import datetime, timedelta
from time import sleep
from pathlib import Path
from picamzero import Camera
import cv2
import math

# =============================================================================
# Utility Functions with Exception Handling and Documentation
# =============================================================================

def get_time(image):
    """
    Extracts the original date and time from the image's EXIF data.

    Args:
        image (str): The file path of the image.

    Returns:
        datetime: The datetime object representing the image's capture time.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the EXIF data does not contain a valid 'datetime_original'.
        Exception: For any other issues during file reading or parsing.
    """
    try:
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get("datetime_original")
            if not time_str:
                raise ValueError(f"EXIF data does not contain 'datetime_original' for {image}")
            time_value = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Error in get_time for image '{image}': {e}")
        raise
    return time_value


def get_time_difference(image_1, image_2):
    """
    Calculates the absolute time difference between two images based on their EXIF datetime.

    Args:
        image_1 (str): File path for the first image.
        image_2 (str): File path for the second image.

    Returns:
        float: The time difference in seconds.

    Raises:
        Exception: Propagates exceptions from get_time if any occur.
    """
    try:
        time_1 = get_time(image_1)
        time_2 = get_time(image_2)
        time_difference = time_2 - time_1
    except Exception as e:
        print(f"Error in get_time_difference: {e}")
        raise
    return abs(time_difference.total_seconds())


def convert_to_cv(image_1, image_2):
    """
    Converts images to grayscale using OpenCV.

    Args:
        image_1 (str): File path for the first image.
        image_2 (str): File path for the second image.

    Returns:
        tuple: A tuple containing the two images in grayscale (numpy arrays).
    
    Raises:
        ValueError: If an image cannot be read.
    """
    image_1_cv = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
    image_2_cv = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)
    if image_1_cv is None:
        raise ValueError(f"Failed to read image: {image_1}")
    if image_2_cv is None:
        raise ValueError(f"Failed to read image: {image_2}")
    return image_1_cv, image_2_cv


def calculate_features(image_1, image_2, feature_number):
    """
    Detects and computes ORB features for both images.

    Args:
        image_1 (ndarray): The first grayscale image.
        image_2 (ndarray): The second grayscale image.
        feature_number (int): Maximum number of features to detect.

    Returns:
        tuple: (keypoints_1, keypoints_2, descriptors_1, descriptors_2) for the two images.
    
    Raises:
        Exception: If ORB detection fails.
    """
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    """
    Matches descriptors from two images using the brute-force matcher.

    Args:
        descriptors_1: Descriptors from the first image.
        descriptors_2: Descriptors from the second image.

    Returns:
        list: A list of matches sorted by distance.
    
    Raises:
        ValueError: If either descriptor set is None.
    """
    if descriptors_1 is None or descriptors_2 is None:
        raise ValueError("Descriptors cannot be None for matching.")
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    """
    Retrieves matching keypoint coordinates from two sets of keypoints using provided matches.

    Args:
        keypoints_1: Keypoints from the first image.
        keypoints_2: Keypoints from the second image.
        matches: List of cv2.DMatch objects.

    Returns:
        tuple: Two lists containing the matching coordinates from both images.
    """
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        try:
            image_1_idx = match.queryIdx
            image_2_idx = match.trainIdx
            (x1, y1) = keypoints_1[image_1_idx].pt
            (x2, y2) = keypoints_2[image_2_idx].pt
            coordinates_1.append((x1, y1))
            coordinates_2.append((x2, y2))
        except IndexError as e:
            print(f"Index error in find_matching_coordinates: {e}")
            continue
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    """
    Calculates the mean Euclidean distance between corresponding coordinates.

    Args:
        coordinates_1: List of (x, y) tuples for the first image.
        coordinates_2: List of (x, y) tuples for the second image.

    Returns:
        float: The average distance between matching features in pixels.
    """
    if not coordinates_1 or not coordinates_2 or len(coordinates_1) != len(coordinates_2):
        return 0.0
    total_distance = 0.0
    for (x1, y1), (x2, y2) in zip(coordinates_1, coordinates_2):
        distance = math.hypot(x1 - x2, y1 - y2)
        total_distance += distance
    return total_distance / len(coordinates_1)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    """
    Calculates the speed in kilometers per second.

    Args:
        feature_distance (float): The average feature distance in pixels.
        GSD (float): Ground Sample Distance value.
        time_difference (float): Time difference in seconds.

    Returns:
        float: Speed in kilometers per second.
    
    Raises:
        ValueError: If time_difference is zero.
    """
    if time_difference == 0:
        raise ValueError("Time difference is zero; cannot compute speed.")
    # Convert feature distance in pixels to kilometers using the GSD.
    distance_km = feature_distance * GSD / 100000.0
    speed = distance_km / time_difference
    return speed


# =============================================================================
# Main Program
# =============================================================================

def main():
    """
    Main function to capture images, process them, and calculate the speed of an object.
    
    The program:
        - Captures two consecutive images using the Camera.
        - Reads their EXIF data to determine the capture time.
        - Uses ORB features to find matching keypoints between the two images.
        - Calculates the average feature displacement in pixels.
        - Computes the object's speed (km/s) based on displacement, Ground Sample Distance (GSD),
          and the time difference between captures.
        - Runs for 5 minutes, logging each measurement, and finally writes the average speed
          to 'results.txt'.
    """
    start_time = datetime.now()
    now_time = datetime.now()

    # Determine the base folder and create a results file.
    base_folder = Path(__file__).parent.resolve()
    data_file = base_folder / "results.txt"
    data_file.touch(exist_ok=True)

    cam = Camera()
    i = 1
    speeds = []  # List to store each measured speed

    # Run the image capture and processing loop for 5 minutes.
    while now_time < start_time + timedelta(minutes=5):
        try:
            # Capture the first image.
            image_1 = cam.take_photo(f"image{i}.jpg")
            i += 1
            sleep(9)  # Wait before taking the second image.

            # Capture the second image.
            image_2 = cam.take_photo(f"image{i}.jpg")
            i += 1

            # Calculate the time difference between images using EXIF data.
            time_difference = get_time_difference(image_1, image_2)

            # Convert images to grayscale for OpenCV processing.
            image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)

            # Detect and compute ORB features.
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)

            # Match the descriptors between images.
            matches = calculate_matches(descriptors_1, descriptors_2)

            # Extract matching coordinates from keypoints.
            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

            # Calculate the average displacement (in pixels) between the matching features.
            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

            # Compute speed using the average feature distance, a given GSD, and the time difference.
            speed = calculate_speed_in_kmps(average_feature_distance, 24451.525, time_difference)

            # Save the current speed measurement.
            speeds.append(speed)

            # Print the measurements for this iteration.
            print("Time difference (seconds):", time_difference)
            print("Average feature distance (pixels):", average_feature_distance)
            print("Speed (km/s):", speed)
        except Exception as e:
            print(f"Error during iteration: {e}")
        finally:
            sleep(1)
            now_time = datetime.now()

    # Compute the average speed over all successful iterations.
    if speeds:
        average_speed = sum(speeds) / len(speeds)
    else:
        average_speed = 0

    # Write the average speed to results.txt.
    try:
        with data_file.open("w") as f:
            f.write(str(round(average_speed, 5)))
    except Exception as e:
        print(f"Error writing to file: {e}")

    print("Final Average Speed (km/s):", round(average_speed, 5))


if __name__ == "__main__":
    main()
