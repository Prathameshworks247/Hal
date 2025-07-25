
import cv2
import numpy as np
import csv
import os
import io
import tempfile
import uuid
from typing import Optional
import base64
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse
from pydantic import BaseModel
import shutil

def get_pixels_per_mm(image, aruco_marker_length_mm=50, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """Detect ArUco marker and calculate pixel-to-mm ratio"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if corners:
        c = corners[0][0]
        marker_width_px = np.linalg.norm(c[0] - c[1])
        pixels_per_mm = marker_width_px / aruco_marker_length_mm
        return pixels_per_mm
    else:
        return None

def detect_shapes(image, pixels_per_mm):
    """Detect shapes and extract their dimensions"""
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data = []  

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        # Approximate shape
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        shape_type = "Unknown"
        if len(approx) == 3:
            shape_type = "Triangle"
        elif len(approx) == 4:
            shape_type = "Rectangle"
        elif len(approx) > 6:
            shape_type = "Circle"

        # Bounding box & measurements
        x, y, w, h = cv2.boundingRect(cnt)
        width_mm = round(w / pixels_per_mm, 2)
        height_mm = round(h / pixels_per_mm, 2)
        area_mm = round(area / (pixels_per_mm ** 2), 2)
        perimeter_mm = round(peri / pixels_per_mm, 2)

        # Centroid
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        # Append to data
        data.append({
            "Shape": shape_type,
            "Area (mm^2)": area_mm,
            "Perimeter (mm)": perimeter_mm,
            "Width (mm)": width_mm,
            "Height (mm)": height_mm,
            "Center X (px)": cX,
            "Center Y (px)": cY
        })

        # Draw and annotate
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(output, f"{shape_type}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.putText(output, f"W:{width_mm}mm H:{height_mm}mm", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.putText(output, f"Area:{area_mm}mm^2", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.putText(output, f"Peri:{perimeter_mm}mm", (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    return output, data

def save_to_csv(data, path):
    """Save shape data to CSV file"""
    keys = ["Shape", "Area (mm^2)", "Perimeter (mm)", "Width (mm)", "Height (mm)", "Center X (px)", "Center Y (px)"]
    with open(path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)