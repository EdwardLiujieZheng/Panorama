import cv2
from image_utils import make_panorama

if __name__ == "__main__":
    # make yosemite panorama
    img1 = cv2.imread("data/yosemite1.jpg")
    img2 = cv2.imread("data/yosemite2.jpg")
    img3 = cv2.imread("data/yosemite3.jpg")
    img4 = cv2.imread("data/yosemite4.jpg")

    img5 = make_panorama(img1, img2)
    img6 = make_panorama(img2, img3)
    img7 = make_panorama(img3, img4)

    img8 = make_panorama(img5, img6)
    img9 = make_panorama(img6, img7)

    img10 = make_panorama(img8, img9)
    img10 = img10[:-10,:-350]

    cv2.imwrite("yosemite_panorama.jpg", img10)

    # make my own panorama
    img1 = cv2.imread("data/IMG_4207.jpg")
    img2 = cv2.imread("data/IMG_4208.jpg")
    img3 = cv2.imread("data/IMG_4209.jpg")
    img4 = cv2.imread("data/IMG_4210.jpg")

    img5 = make_panorama(img1, img2)
    img6 = make_panorama(img2, img3)
    img7 = make_panorama(img3, img4)

    img8 = make_panorama(img5, img6)
    img9 = make_panorama(img6, img7)

    img10 = make_panorama(img8, img9)

    img10 = img10[:,:-800]
    cv2.imwrite("my_panorama.jpg", img10)