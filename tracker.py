from random import randint
from vehicle_identification import VehicleIdentificator
import numpy as np

class TrajectoryNode:
    def __init__(self, box, image, emb) -> None:
        self.box = box
        self.image = image
        self.emb = emb

class Trajectory:
    def __init__(self, id, start_node, frame_index) -> None:
        self.id = id
        self.nodes = [start_node]
        self.last_update_frame_index = frame_index
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

    def add(self, node, frame_index):
        self.nodes.append(node)
        self.last_update_frame_index = frame_index

    def mean_emb(self):
        mean_emb = np.zeros(self.nodes[0].emb.shape)
        for node in self.nodes[-10:]:
            mean_emb += node.emb

        return mean_emb / len(self.nodes)

    def is_box_in_trajectory(self, box):
        iou = self.iou(box, self.nodes[-1])
        if iou > 0.5:
            return True

        return False

    def iou(self, boxA, boxB):
        xA = max(boxA.left, boxB.left)
        yA = max(boxA.top, boxB.top)
        xB = min(boxA.right, boxB.right)
        yB = min(boxA.bottom, boxB.bottom)
        interArea = abs((xB - xA) * (yB - yA))
        if interArea == 0:
            return 0
        boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top)
        boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

class Tracker:
    def __init__(self, vehicle_model_path) -> None:
        self.vehicle_identificator = VehicleIdentificator(vehicle_model_path)
        self.trajectories = []
        self.new_id = 0

    def update(self, boxes, frame, frame_index):
        vehicle_images = [frame[box.top:box.bottom, box.left:box.right, :] for box in boxes]
        embs = self.vehicle_identificator.get_embeddings(vehicle_images)

        nodes = []
        for box, img, emb in zip(boxes, vehicle_images, embs):
            nodes.append(TrajectoryNode(box, img, emb))

        trajectories_to_add = []
        for node in nodes:
            if len(self.trajectories) == 0:
                trajectories_to_add.append(Trajectory(self.new_id, node, frame_index)) 
                self.new_id += 1
                continue

            compare_confidences = [self.vehicle_identificator.compare(node.emb, trajectory.mean_emb()) for trajectory in self.trajectories]
            index = np.argmax(np.array(compare_confidences))
            if compare_confidences[index] > 0.5:
                self.trajectories[index].add(node, frame_index)
            else:
                trajectories_to_add.append(Trajectory(self.new_id, node, frame_index))
                self.new_id += 1
        
        for trajectory in trajectories_to_add:
            self.trajectories.append(trajectory)

        active_trajectories = []
        for i, trajectory in enumerate(self.trajectories):
            if not frame_index - trajectory.last_update_frame_index >= 100:
                active_trajectories.append(trajectory)

        self.trajectories = active_trajectories

class BoundingBox:
    def __init__(self, left, top, right, bottom) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom