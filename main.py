# import module
import cv2

# base variable
HEIGHT = 512
WIDTH = 512
MODEL_PATH = "./model/<model_name>/intel/<precision>/<model_name>.xml"

# load xml files
cascade = cv2.CascadeClassifier(MODEL_PATH)

# get cv2 video frame from web cam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (HEIGHT, WIDTH), cv2.INTER_AREA)

    # draw human rect angle

    # draw skelton rect angle

    cv2.imshow('streaming', img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


