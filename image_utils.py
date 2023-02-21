import numpy as np
import cv2
from matplotlib import pyplot as plt

def detect_features(img, show_img=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    key, des = sift.detectAndCompute(img_gray, None)

    if show_img:
        out = cv2.drawKeypoints(img, key, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("features", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return key, des



def match_features(des1, des2, show_img=False):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if show_img:
        try:
            out = cv2.drawMatches(img1, key1, img2, key2, matches[:10], 0, flags=2)
            cv2.imshow("matches", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except NameError:
            print("NameError: cv2.drawMatches requires more arguments!")

    return matches



if __name__ == "__main__":
    img1 = cv2.imread("data/yosemite1.jpg")
    img2 = cv2.imread("data/yosemite2.jpg")

    key1, des1 = detect_features(img1)
    key2, des2 = detect_features(img2)

    matches = match_features(des1, des2)

    print(matches[0].queryIdx, matches[0].trainIdx)
    print(key1[matches[0].queryIdx].pt, key2[matches[0].trainIdx].pt)