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
    markup_path = os.path.join('test_video', f'test_video_{video_index}.csv')
    target_image_path = os.path.join('samples', f'sample{video_index}.png')


    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    data = read_markup(markup_path, (width, height))
 
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    target_image = cv2.imread(target_image_path)
    tracker = Tracker("reidentification.onnx")
    vehicle_identificator = VehicleIdentificator("reidentification.onnx")
    target_emb = vehicle_identificator.get_embeddings([target_image])[0]

    print("Gathering data...")
    frame_index = 0
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame_index in data:
            tracker.update(data[frame_index], frame, frame_index)
        if ret == True:
            for trajectory in tracker.trajectories:
                if vehicle_identificator.compare(trajectory.mean_emb(), target_emb) > 0.5:
                    if trajectory.last_update_frame_index == frame_index:
                        frames.append(cv2.resize(trajectory.nodes[-1].image, (416, 416)))
        else:
            break
        frame_index += 1
    
    if not os.path.exists("task2_output"):
        os.mkdir("task2_output")
    output_dir = os.path.join("task2_output", f"{video_index}")
    for i, frame in enumerate(frames):
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