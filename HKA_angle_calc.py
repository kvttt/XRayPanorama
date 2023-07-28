import math
import cv2
import sys
import pydicom

# python HKA_angle_calc.py composite_file

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    angle = math.degrees(
        math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
        math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    return angle + 360 if angle < 0 else angle

def main():
    # Load image
#    img = cv2.imread("IM-0001-0001.jpg")
    dcm = pydicom.dcmread(sys.argv[1])
    img = dcm.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
# Select three points
    points = []
    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
            if len(points) == 3:
                angle = calculate_angle(*points)
                print("The HKA Angle is:", 180-angle)
                cv2.line(img, points[0], points[1], (0, 255, 0), 2)
                cv2.line(img, points[1], points[2], (0, 255, 0), 2)
                cv2.imshow("image", img)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
            

    cv2.imshow("image", img)
    cv2.setMouseCallback("image", select_point)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
