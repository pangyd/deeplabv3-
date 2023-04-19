import cv2
import numpy as np
import data


def change_channels(changed, i):
    # Load image
    img = cv2.imread('D://deeplabv3-plus-pytorch-main//datasets//fireimg//fire07_113.jpg')
    cv2.imshow("Origin", img)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, changed)

    # Split channels
    l, a, b = cv2.split(lab)

    pts = np.array(data.data, np.int32)
    rgb_masked = []

    # Create mask
    # for x in [l, a, b]:
    mask = np.zeros_like(lab)
    l1, a1, b1 = cv2.split(mask)

    for x, y in zip([l, a, b], [l1, a1, b1]):
        cv2.fillPoly(y, [pts], (20, 20, 20))

        masked = cv2.bitwise_or(x, y)
        rgb_masked.append(masked)

    # Merge channels
    lab = cv2.merge((rgb_masked[0], rgb_masked[1], rgb_masked[2]))

    # Convert back to BGR color space
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Display result
    cv2.imshow('Result{}'.format(i), result)

# changed = cv2.COLOR_BGR2LAB
# change_channels(changed, 1)
# change_channels(cv2.COLOR_BGR2YCrCb, 2)
# change_channels(cv2.COLOR_BGR2HSV, 3)
# change_channels(cv2.COLOR_BGR2HLS, 4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def change_pointval():
    # Load image
    img = cv2.imread('D://deeplabv3-plus-pytorch-main//datasets//fireimg//fire07_113.jpg')
    cv2.imshow("Origin", img)

    # Define polygon points
    # pts = np.array([[10, 10], [700, 10], [700, 700], [10, 700]], np.int32)
    pts = np.array(data.data, np.int32)

    # Create mask
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [pts], (100, 100, 100))

    # Apply mask to image
    masked_img = cv2.bitwise_or(img, mask)

    # Perform pixel-level processing on masked image
    # ...

    # Display result
    cv2.imshow('Result', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# change_pointval()






