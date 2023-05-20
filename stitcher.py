import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

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



def warp_img(img1, key1, img2, key2, matches, show_img=False):
    pts1 = np.float32([key1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([key2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    img2_warped = cv2.warpPerspective(img2, H, (int(img1.shape[1] * 1.5), img1.shape[0]))

    if show_img:
        cv2.imshow("warped", img2_warped)
        img1_larger = np.zeros((img1.shape[0], img1.shape[1] * 2, 3), np.uint8)
        img1_larger[:img1.shape[0], :img1.shape[1]] = img1.copy()
        out = cv2.addWeighted(img1_larger, 0.5, img2_warped, 0.5, 0)
        cv2.imshow("combined", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img2_warped


def get_distance_transform(img_rgb):
    """
    Get distance to the closest background pixel for an RGB image
    Input:
    img_rgb: np.array , HxWx3 RGB image
    Output:
    dist: np.array , HxWx1 distance image
    each pixel’s intensity is proportional to
    its distance to the closest background pixel
    scaled to [0..255] for plotting
    """
    # Threshold the image: any value above 0 maps into 255
    thresh = cv2.threshold(img_rgb , 0, 255, cv2.THRESH_BINARY)[1]
    # Collapse the color dimension
    thresh = thresh.any(axis =2)
    # Pad to make sure the border is treated as background
    thresh = np.pad(thresh , 1)
    # Get distance transform
    dist = distance_transform_edt(thresh)[1:-1, 1:-1]
    # HxW -> HxWx1
    dist = dist[:, :, None]
    return dist / dist.max() * 255.0



def blend_img(img1, img2, show_alpha=False, show_img=False):
    img1_larger = np.zeros((img1.shape[0], int(img1.shape[1] * 1.5), 3), np.uint8)
    img1_larger[:img1.shape[0], :img1.shape[1]] = img1.copy()
    alpha = get_distance_transform(img1_larger) / 255.0
    alpha[:, :int(img1.shape[1] * 2 / 3)] = 1
    if show_alpha:
        cv2.imshow("alpha", alpha)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    out = (img1_larger * alpha + img2 * (1 - alpha)) / 255.0
    if show_img:
        cv2.imshow("out", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out



def stitch(img1, img2, show_img=False):
    key1, des1 = detect_features(img1)
    key2, des2 = detect_features(img2)
    matches = match_features(des1, des2)
    img2_warped = warp_img(img1, key1, img2, key2, matches)
    out = blend_img(img1, img2_warped, False, show_img)
    out = (out * 255.0).astype(np.uint8)
    return out



