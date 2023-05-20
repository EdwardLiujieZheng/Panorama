import os
import argparse
import cv2
from stitcher import stitch


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/images', help='directory of the input images')
parser.add_argument('--output', type=str, default='data/out/panorama.jpg', help='directory and name of the output panorama')
parser.add_argument('--crop', type=tuple, default=(0, 10, 0, 400), help='crop the output by (up, down, left, right)')
args = parser.parse_args()

# make panorama
imgs_all_steps = [[]]
extensions = ('.jpg', '.png')
for filename in sorted(os.listdir(args.input)):
    if filename.endswith(extensions):
        img = cv2.imread(os.path.join(args.input, filename))
        imgs_all_steps[0].append(img)

step = 0
while len(imgs_all_steps[step]) > 1:
    print(f"Stitching in progress -- step {step + 1}")
    imgs = []
    for img1, img2 in zip(imgs_all_steps[step][:-1], imgs_all_steps[step][1:]):
        imgs.append(stitch(img1, img2))
    imgs_all_steps.append(imgs)
    step += 1

# crop and write
out = imgs_all_steps[-1][0]
out = out[args.crop[0]:-args.crop[1], args.crop[2]:-args.crop[3]]
cv2.imwrite(args.output, out)

print("Done!")