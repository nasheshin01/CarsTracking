import cv2
import sys
import os

from tracker import Tracker
from utils import read_markup

def draw_trajectories(frame, trajectories, frame_index):
    for trajectory in trajectories:
        if frame_index != trajectory.last_update_frame_index:
            continue

        last_box = trajectory.nodes[-1].box
        cv2.rectangle(frame, (last_box.left, last_box.top), (last_box.right, last_box.bottom), trajectory.color, 3)
        cv2.putText(frame, str(trajectory.id), (last_box.left, last_box.top), cv2.FONT_ITALIC, 1, trajectory.color, 4)

    return frame


def main():
    video_index = int(sys.argv[1])
    if video_index > 6 or video_index < 1:
        print("Video index should be from 1 to 6")
        return

    video_path = os.path.join('test_video', f'test_video_{video_index}.mkv')
    markup_path = os.path.join('test_video', f'test_video_{video_index}.csv')

    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    data = read_markup(markup_path, (width, height))
 
    if (cap.isOpened() == False): 
        print("Error opening video stream or file")
        return
    
    tracker = Tracker("reidentification.onnx")

    frame_index = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame_index in data:
            tracker.update(data[frame_index], frame, frame_index)
        frame = draw_trajectories(frame, tracker.trajectories, frame_index)
        frame_index += 1
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()