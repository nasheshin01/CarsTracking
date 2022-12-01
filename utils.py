import pandas as pd
from tracker import BoundingBox

def read_markup(file_path, video_resolution):
    df = pd.read_csv(file_path, header=0)

    width = video_resolution[0]
    height = video_resolution[1]
    markup_dict = {}
    for frame_index in df['frame'].unique():
        boxes = df[df['frame'] == frame_index].values[:,1:]
        bounding_boxes = []
        for box in boxes:
            box_left = int(box[0] * width)
            box_top = int(box[1] * height)
            box_right = int(box[2] * width)
            box_bottom = int(box[3] * height)
            bounding_boxes.append(BoundingBox(box_left, box_top, box_right, box_bottom))

        markup_dict[frame_index] = bounding_boxes
        

    return markup_dict