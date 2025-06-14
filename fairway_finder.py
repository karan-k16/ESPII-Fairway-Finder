import cv2
import numpy as np
import json
import glob
import os
import tkinter as tk
from PIL import Image, ImageTk

def load_reference_offset(json_path):
    """
    Loads the reference offset from a JSON file.
    The JSON file should contain "displacement_m" with x and y in meters.
    Returns the offset as a tuple (x, y) in meters.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            dx = data["displacement_m"]["x"]
            dy = data["displacement_m"]["y"]
        return (dx, dy)
    except Exception as e:
        print("Error loading reference offset:", e)
        return (0, 0)

def calibrate_camera(calib_images_folder, chessboard_size=(9,6), square_size=25.0):
    """
    Uses chessboard photos to figure out the camera settings.
    This helps correct any lens distortion so our images look natural.
    """
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(calib_images_folder, '*jpg'))
    if not images:
        print("Couldn't find any calibration images in:", calib_images_folder)
        return None, None, None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        else:
            print("Couldn't detect chessboard in image:", fname)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera calibration finished. Reprojection error:", ret)
    return camera_matrix, dist_coeffs, ret

def compute_homography_from_points(image_points, world_points):
    """
    Creates a mapping (homography matrix) between points in the image and real-life positions.
    This helps translate a point from the photo into a real-world coordinate.
    """
    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    H, status = cv2.findHomography(image_points, world_points)
    return H

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Corrects the image so it appears as if taken with an ideal lens.
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    return undistorted

def detect_golf_ball(image):
    """
    Scans the image for round shapes (possible golf balls or LED lights) using a circle-finding method.
    First, an HSV mask is applied to isolate grey tones (our ball's color), then Hough Circles is used.
    Returns a list of detected circles [x, y, r] (in pixels), sorted by radius (smallest first).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_grey = np.array([0, 0, 80])
    upper_grey = np.array([179, 50, 200])
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=20, minRadius=5, maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = sorted(circles, key=lambda c: c[2])
        return circles
    else:
        return None

def apply_homography(point, H):
    """
    Converts a point from the image (in pixels) to a real-world coordinate using the homography matrix.
    """
    point_homog = np.array([point[0], point[1], 1], dtype=np.float32)
    world_point_homog = H @ point_homog
    world_point_homog /= world_point_homog[2]
    return (world_point_homog[0], world_point_homog[1])

def show_result_gui(undistorted_image, ball_center, radius, radius_m, relative_coords_m):
    """
    Opens a window showing the undistorted image with the detected ball,
    along with text displaying the displacement (in meters) and the ball's size in pixels and meters.
    Uses a canvas with panning (via mouse dragging) so you can view the entire landscape image.
    """
    root = tk.Tk()
    root.title("Fairway Finder Results")
    root.geometry("1600x600")

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)

    def start_pan(event):
        canvas.scan_mark(event.x, event.y)
    def do_pan(event):
        canvas.scan_dragto(event.x, event.y, gain=1)
    canvas.bind("<ButtonPress-1>", start_pan)
    canvas.bind("<B1-Motion>", do_pan)

    rgb_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    img_tk = ImageTk.PhotoImage(master=root, image=pil_img)
    label_img = tk.Label(canvas, image=img_tk)
    label_img.image = img_tk
    canvas.create_window(0, 0, anchor="nw", window=label_img)
    root.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    text_frame = tk.Frame(root)
    text_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    label_disp = tk.Label(text_frame, text=f"Displacement (m): x = {relative_coords_m[0]:.3f}, y = {relative_coords_m[1]:.3f}")
    label_disp.pack(side=tk.LEFT, padx=10)
    label_radius = tk.Label(text_frame, text=f"Detected Radius: {radius} px ({radius_m:.4f} m)")
    label_radius.pack(side=tk.LEFT, padx=10)
    button_close = tk.Button(text_frame, text="Close", command=root.destroy)
    button_close.pack(side=tk.RIGHT, padx=10)

    root.mainloop()

def process_image(input_image_path, camera_matrix, dist_coeffs, H, output_json_path, starting_world_coords, conversion_factor, is_reference=False):
    """
    Processes the Blender-rendered image:
      - Loads and undistorts the image.
      - Detects the golf ball using color filtering and circle detection.
      - Converts the detected ball's position from pixels to world coordinates (in mm), then to meters.
      - Computes the displacement relative to the provided reference offset.
      - Saves the results as JSON.
    """
    image = cv2.imread(input_image_path)
    if image is None:
        print("Couldn't load the image:", input_image_path)
        return None, None, None, None, None

    print("Loaded image shape:", image.shape)
    undistorted = undistort_image(image, camera_matrix, dist_coeffs)
    circles = detect_golf_ball(undistorted)
    if circles is None:
        print("Could not detect any circular objects!")
        return None, None, None, None, None

    expected_ball_radius_m = 0.045  # Expected golf ball radius ~4.5 cm.
    selected_circle = None
    min_diff = float('inf')
    for circle in circles:
        radius_px = circle[2]
        radius_m_candidate = radius_px * conversion_factor
        diff = abs(radius_m_candidate - expected_ball_radius_m)
        if diff < min_diff:
            min_diff = diff
            selected_circle = circle

    if selected_circle is None:
        print("No suitable golf ball detected!")
        return None, None, None, None, None

    ball_center = (selected_circle[0], selected_circle[1])
    radius = selected_circle[2]
    radius_m = radius * conversion_factor

    world_coords = apply_homography(ball_center, H)  # world_coords in mm.
    world_coords_m = (world_coords[0] / 1000.0, world_coords[1] / 1000.0)  # in meters
    print("Raw world coordinates (m):", world_coords_m)
    
    # Compute the relative displacement (in m) relative to the reference offset.
    relative_coords_m = (world_coords_m[0] - starting_world_coords[0],
                         world_coords_m[1] - starting_world_coords[1])
    
    # If this is the reference image, you may choose to output its raw coordinates.
    # Here, we output raw coordinates for the reference (you can force (0,0) if desired).
    if is_reference:
        relative_coords_m = world_coords_m

    print("Reference offset (m):", starting_world_coords)
    print("Computed relative displacement (m):", relative_coords_m)

    output_data = {
        "displacement_m": {
            "x": float(relative_coords_m[0]),
            "y": float(relative_coords_m[1])
        },
        "radius_pixels": int(radius),
        "radius_m": float(radius_m)
    }
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print("Output saved to", output_json_path)

    cv2.circle(undistorted, ball_center, radius, (0, 255, 0), 2)
    cv2.circle(undistorted, ball_center, 2, (0, 0, 255), 3)

    return undistorted, ball_center, radius, relative_coords_m, radius_m

def main():
    print("Starting Fairway Finder (Blender Model Version)...")
    calib_folder = "calib_images"
    chessboard_size = (9, 6)
    square_size = 25.0  # Size of each chessboard square in mm.
    if os.path.exists(calib_folder):
        print("Calibrating the camera...")
        camera_matrix, dist_coeffs, _ = calibrate_camera(calib_folder, chessboard_size, square_size)
    else:
        print("No calibration images found. Using default settings.")
        camera_matrix = np.array([[1000, 0, 640],
                                  [0, 1000, 360],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5,), dtype=np.float32)

    # Updated image_points based on your manual clicks:
    image_points = [
        [99, 638],    # Top-left
        [3740, 639],  # Top-right
        [3743, 1528], # Bottom-right
        [102, 1530]   # Bottom-left
    ]
    # Update world_points to match your green's physical dimensions: 4.88 m x 1.22 m (in mm)
    world_points = [
        [0, 0],         # Top-left in real-world coordinates.
        [4880, 0],      # Top-right (4.88 m)
        [4880, 1220],   # Bottom-right (4.88 m, 1.22 m)
        [0, 1220]       # Bottom-left (1.22 m)
    ]
    H = compute_homography_from_points(image_points, world_points)
    print("Calculated Homography Matrix:")
    print(H)

    # Process the reference image first.
    reference_json = "reference_output.json"
    reference_image_path = "reference_image.jpg"  # Replace with your reference image filename.
    undistorted_ref, ball_center_ref, radius_ref, ref_disp_m, radius_m_ref = process_image(
        reference_image_path, camera_matrix, dist_coeffs, H, reference_json, (0, 0),
        ((world_points[1][0]-world_points[0][0])/(image_points[1][0]-image_points[0][0]))/1000.0,
        is_reference=True
    )
    if undistorted_ref is not None:
        print("Reference image processed. Saved reference offset to", reference_json)
    else:
        print("Reference image processing failed.")

    # Load the reference offset from the JSON file.
    starting_world_coords = load_reference_offset(reference_json)
    print("Using starting_world_coords (reference offset):", starting_world_coords)
    
    conversion_factor = ((world_points[1][0]-world_points[0][0])/(image_points[1][0]-image_points[0][0]))/1000.0
    print("Conversion factor (meters per pixel):", conversion_factor)

    # List of subsequent images to process.
    subsequent_images = ["blender_model6.jpg", "blender_model5.jpg"]

    for img_file in subsequent_images:
        output_json_path = f"output_{os.path.splitext(os.path.basename(img_file))[0]}.json"
        undistorted, ball_center, radius, relative_coords_m, radius_m = process_image(
            img_file, camera_matrix, dist_coeffs, H, output_json_path, starting_world_coords, conversion_factor, is_reference=False)
        if undistorted is not None:
            show_result_gui(undistorted, ball_center, radius, radius_m, relative_coords_m)
        else:
            print(f"Image processing failed for {img_file}")

    print("Process completed.")

if __name__ == "__main__":
    main()
