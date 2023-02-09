import cv2
import pafy

# キャプチャするYouTube動画のURLを設定します
url = ""

video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url,)

while (True):
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()

# install pafy and youtube_dl module

# comment out backend_youtube_dl.py line 53, 54
