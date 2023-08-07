import cv2
import pydicom as dicom

import math
import sys

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""

    angle = math.degrees(
        math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
        math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    )

    return angle + 360 if angle < 0 else angle

dcm = dicom.dcmread(sys.argv[1])
composite_array = dcm.pixel_array

cv2.destroyAllWindows()

composite_array = (composite_array - composite_array.min()) / (composite_array.max() - composite_array.min())

# Select three points
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP:
        if len(points) <= 2:
            points.append((x, y))
            print(f'Point {len(points)} selected at ({x}, {y}).')
            cv2.circle(composite_array, (x, y), 30, (0, 0, 255), -1)
            cv2.imshow('Composite', composite_array)
            if len(points) == 3:
                cv2.line(composite_array, points[0], points[1], (0, 255, 0), 20, cv2.LINE_AA)
                cv2.line(composite_array, points[1], points[2], (0, 255, 0), 20, cv2.LINE_AA)
                cv2.imshow('Composite', composite_array)
                print('HKA angle specified. Calculating the HKA angle...')
                angle = calculate_angle(*points)
                print(f'The HKA angle is: {180 - angle}. Press Enter to exit.')
                if cv2.waitKey(20) == ord('\r'):
                    cv2.destroyAllWindows()

# initialize window
cv2.namedWindow('Composite', cv2.WINDOW_NORMAL)
cv2.imshow('Composite', composite_array)
cv2.resizeWindow('Composite',
                 int(composite_array.shape[1] / 10),
                 int(composite_array.shape[0] / 10),
                 )
cv2.setMouseCallback('Composite', select_point)
print('Please click three times to specify three points. Press Enter to confirm...')
cv2.waitKey()
cv2.destroyAllWindows()
