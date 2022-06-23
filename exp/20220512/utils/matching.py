from .visualize import get_masked_poses, get_masked_bboxes, tlwh2tldr
import numpy as np
from ipdb import set_trace as ist


def get_pose_bbox_position(pose):
    """姿勢のbbox領域を獲得する
    Args:
        pose list(float): キーポイントの座標[[float],....]
    Returns:
        top_left, down_right: 左上(x,y), 右上(x,y)
    """
    INF = 1 << 20
    min_x, max_x = INF, -1
    min_y, max_y = INF, -1

    for x, y, _ in pose:
        x, y = float(x), float(y)
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    left_top = (min_x, min_y)
    right_down = (max_x, max_y)

    return left_top, right_down


def calc_IOU_score(a, b) -> float:
    """IOUを計算する
    Args:
        a: righttop, leftdown
        b: righttop, leftdown
    Returns:
        IoU score: float
    """

    (min_ax, min_ay), (max_ax, max_ay) = a[0], a[1]
    (min_bx, min_by), (max_bx, max_by) = b[0], b[1]

    a_area = (max_ax - min_ax + 1) * (max_ay - min_ay + 1)
    b_area = (max_bx - min_bx + 1) * (max_by - min_by + 1)

    inner_min_x = max(min_ax, min_bx)
    inner_max_x = min(max_ax, max_bx)
    inner_min_y = max(min_ay, min_by)
    inner_max_y = min(max_ay, max_by)

    w = max(0, inner_max_x - inner_min_x + 1)
    h = max(0, inner_max_y - inner_min_y + 1)
    intersect = w * h

    iou = intersect / (a_area + b_area - intersect)

    return iou


def match_id_by_iou(poses, bboxes, frame_id: int) -> list:
    """kapaoのidとdeepsortのidをくっつける
    Args:
        poses: kapaoのデータ
        bboxes: deepsortのデータ
        frame_id: 対象のフレーム
    Returns:
        deepsortのidと同じになったposeデータ
    """
    poses = get_masked_poses(poses, frame_id)
    bboxes = get_masked_bboxes(bboxes, frame_id)

    if len(bboxes) == 0:
        return poses

    out_poses = []
    for pose in poses:
        _frame_id, _, pose = pose
        pose_topleft, pose_rightdown = get_pose_bbox_position(pose)

        scores: list = []
        for bbox in bboxes:
            bbox = bbox[2:6]
            bbox = tlwh2tldr(bbox)
            bbox_topleft, bbox_rightdown = (bbox[0], bbox[1]), (bbox[2], bbox[3])
            score = calc_IOU_score(
                (pose_topleft, pose_rightdown), (bbox_topleft, bbox_rightdown)
            )
            scores.append(score)

        scores = np.array(scores)
        max_idx = np.argmax(scores)

        matched_tracklet = bboxes[max_idx][1] if scores[max_idx] > 0.2 else -1
        matched_pose_data = [_frame_id, matched_tracklet, pose]
        out_poses.append(matched_pose_data)
    return out_poses
