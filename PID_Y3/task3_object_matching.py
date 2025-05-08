import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tkinter as tk
from tkinter import filedialog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory containing the .py file
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_object(object_img_path, scene_img_path):
    # Read images
    object_img = cv2.imread(object_img_path)
    scene_img = cv2.imread(scene_img_path)
    
    # Check if images were successfully loaded
    if object_img is None:
        print(f"Error: Could not read object image from '{object_img_path}'")
        return None, None
    if scene_img is None:
        print(f"Error: Could not read scene image from '{scene_img_path}'")
        return None, None
    
    # Convert to grayscale
    object_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(object_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)
    
    # Create BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher()
    
    # Find top 2 matches for each descriptor
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # Ensure enough matches to find homography
    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        # Extract locations of matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix using RANSAC
        # M is the homography matrix, mask is the status mask returned from RANSAC
        # mask contains 0 for outliers and 1 for inliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Use only inliers (points selected by RANSAC)
        # mask.ravel() converts 2D mask to 1D array
        # matchesMask is used to distinguish inliers (1) and outliers (0) during drawing
        matchesMask = mask.ravel().tolist()
        
        # Get height and width of object image
        h, w = object_gray.shape
        
        # Define 4 corners of the object in the original image
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # Transform the 4 corners to corresponding positions in the scene image
        # Use homography matrix M to map points from object image to scene image
        dst = cv2.perspectiveTransform(pts, M)
        
        # Draw bounding box around object in scene image
        scene_with_box = scene_img.copy()
        # polylines draws a polygon connecting the points (4 corners found) with a green outline
        cv2.polylines(scene_with_box, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Prepare image to display matched points
        # Use matchesMask to distinguish inliers/outliers by color
        draw_params = dict(
            matchColor=(0, 255, 0),  # green color for matches (inliers)
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers based on mask from findHomography
            flags=2
        )
        
        # Draw good matched points (inliers)
        match_img = cv2.drawMatches(cv2.copyMakeBorder(cv2.resize(object_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR), 0, 0, 0, 150, cv2.BORDER_CONSTANT, value=(0,0,0)), kp1, scene_img, kp2, good_matches, None, **draw_params)

        # Convert from BGR to RGB for display with matplotlib
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        scene_with_box_rgb = cv2.cvtColor(scene_with_box, cv2.COLOR_BGR2RGB)
        
        # Display results
        plt.figure(figsize=(150, 10))
        
        plt.subplot(211)
        plt.imshow(match_img_rgb)
        plt.title('Matched Points (Inliers)', fontsize=14)
        plt.axis('off')
        
        plt.subplot(212)
        plt.imshow(scene_with_box_rgb)
        plt.title('Detected Object', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Create result filename from input filenames
        result_file = os.path.join(OUTPUT_DIR, f"result_task3.jpg")
        
        plt.savefig(result_file, dpi=300)
        plt.show()
        
        return scene_with_box, result_file
    
    else:
        print(f"Not enough matches - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None, None

def select_image_dialog(title="Select image file"):
    """Display file dialog and return selected file path"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

if __name__ == "__main__":
    # Create parser for command-line arguments
    parser = argparse.ArgumentParser(description='Find and locate object in scene image using SIFT')
    parser.add_argument('--object', type=str, help='Path to object image to find')
    parser.add_argument('--scene', type=str, help='Path to scene image containing object')
    
    args = parser.parse_args()
    
    # Determine input image paths
    object_img_path = None
    scene_img_path = None
    
    # If command-line arguments are provided, use them
    if args.object and args.scene:
        object_img_path = args.object
        scene_img_path = args.scene
    else:
        # If no command-line arguments, use GUI by default
        print("Select object image:")
        object_img_path = select_image_dialog("Select object image")
        if not object_img_path:
            print("No object image selected. Exiting.")
            exit()
            
        print("Select scene image:")
        scene_img_path = select_image_dialog("Select scene image")
        if not scene_img_path:
            print("No scene image selected. Exiting.")
            exit()
    
    print(f"Object image: {object_img_path}")
    print(f"Scene image: {scene_img_path}")
    
    # Perform object detection
    result, result_file = find_object(object_img_path, scene_img_path)
    
    if result is not None:
        # Save result image as required
        # Create result filename from input filenames
        object_name = os.path.splitext(os.path.basename(object_img_path))[0]
        scene_name = os.path.splitext(os.path.basename(scene_img_path))[0]
        output_file = f"scene_{scene_name}_with_detected_{object_name}.jpg"
        
        output_file = os.path.join(OUTPUT_DIR, f"result_task3.jpg")
        cv2.imwrite(output_file, result)
    else:
        print("Object not found in scene image.")