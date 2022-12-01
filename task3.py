import cv2
import sys
import os

from tracker import Tracker
from vehicle_identification import VehicleIdentificator
from utils import read_markup
from dublicate_cleaner import DublicateCleaner


def main():
    video_index = int(sys.argv[1])
    if video_index > 6 or video_index < 1:
        print("Video index should be from 1 to 6")
        return

    video_path = os.path.join('test_video', f'test_video_{video_index}.mkv')

    cap = cv2.VideoCapture(video_path)
 
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    
    cleaner = DublicateCleaner()
    cleaned_frames = cleaner.clean(frames)
    print("Total frames count:", len(frames))
    print("Unique frames count:", len(cleaned_frames))

    if not os.path.exists("task3_output"):
        os.mkdir("task3_output")
    output_dir = os.path.join("task3_output", f"{video_index}")
    for i, frame in enumerate(cleaned_frames):
        cv2.imshow("Frame", frame)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        cv2.imwrite(os.path.join(output_dir, f"{i}.png"), frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()