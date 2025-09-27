import threading
import time
import cv2

class VideoPlayer:
    def __init__(self, source, size=(760, 760), flip=False, skip_first_frames=0):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open {'camera' if isinstance(source, int) else ''} {source}")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._flip = flip
        self._size = size
        self._interpolation = (
            cv2.INTER_AREA
            if size[0] < self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) else cv2.INTER_LINEAR
        )
        _, self._frame = self._cap.read()
        self._lock = threading.Lock()

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join()
        self._cap.release()

    def _run(self):
        prev_time = 0
        while not self._stop:
            t1 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                break

            if 1 / self._fps < time.time() - prev_time:
                prev_time = time.time()
                with self._lock:
                    self._frame = frame

            t2 = time.time()
            wait_time = 1 / self._fps - (t2 -t1)
            time.sleep(max(0, wait_time))
        self._frame = None

    def next(self):
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()

        if self._size is not None:
            frame = cv2.resize(frame, self._size, interpolation=self._interpolation)
        if self._flip:
            frame = cv2.flip(frame, 1)
        return frame
