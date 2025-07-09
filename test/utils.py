import os
import cv2
import numpy as np


def letterbox(
    im: np.ndarray,
    new_shape=(640, 640),
    color=(0, 0, 0),
    scaleup: bool = True,
    reuse_out: np.ndarray = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Resize and pad image to `new_shape` (h, w) with minimal allocations.
    Returns: (out, (r_w, r_h), (dw, dh))
      - out:      resized+padded image
      - ratio:    (width_ratio, height_ratio)
      - padding:  (dw, dh) in float (total padding per axis, before split)
    Args:
      im:         H×W×C uint8 input
      new_shape:  target (height, width)
      color:      padding color
      scaleup:    allow upscaling (False will only downscale)
      reuse_out:  optional buffer of shape (new_h, new_w, C) to reuse
    """
    orig_h, orig_w = im.shape[:2]
    target_h, target_w = new_shape

    # 1. 计算缩放比例
    r = min(target_h / orig_h, target_w / orig_w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad_w, new_unpad_h = int(round(orig_w * r)), int(round(orig_h * r))
    dw, dh = target_w - new_unpad_w, target_h - new_unpad_h
    dw /= 2  # padding on left/right
    dh /= 2  # padding on top/bottom

    # 2. 预分配输出画布（可复用）
    if reuse_out is None or reuse_out.shape[:2] != (target_h, target_w):
        if im.ndim == 3:
            out = np.full((target_h, target_w, im.shape[2]), color, dtype=im.dtype)
        else:
            out = np.full((target_h, target_w), color[0], dtype=im.dtype)
    else:
        out = reuse_out
        # 先填充背景色
        out[:] = color

    # 3. 缩放并写入输出
    if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(
            im, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = im

    # 4. 计算放置位置并原地赋值
    top = int(round(dh))
    left = int(round(dw))
    out[top : top + new_unpad_h, left : left + new_unpad_w] = resized

    return out, (r, r), (dw, dh)


import numpy as np

# --------------------- 配置 ---------------------
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640
DEFAULT_BRANCH = 3


# ------------------- 工具函数 -------------------
def dfl_tensor(position: np.ndarray) -> np.ndarray:
    """
    Vectorized Distribution Focal Loss decoding.
    position: (C=4*mc, H, W)
    返回: (4, H, W)
    """
    p_num = 4
    C, H, W = position.shape
    mc = C // p_num
    # reshape to (4, mc, H, W)
    y = position.reshape(p_num, mc, H, W)
    # subtract max for numerical stability
    y_max = y.max(axis=1, keepdims=True)  # (4,1,H,W)
    e = np.exp(y - y_max)  # (4,mc,H,W)

    # compute expected value: sum(k * e) / sum(e)
    k = np.arange(mc, dtype=np.float32).reshape(1, mc, 1, 1)
    num = (e * k).sum(axis=1)  # (4,H,W)
    den = e.sum(axis=1)  # (4,H,W)
    return num / den  # (4,H,W)


def box_process(position: np.ndarray) -> np.ndarray:
    """
    从 DFL 输出的 position 生成 (H*W, 4) 格式的 [x1,y1,x2,y2]
    position: (C, H, W)
    """
    C, H, W = position.shape
    # grid coords
    col, row = np.meshgrid(np.arange(W), np.arange(H))  # each (H,W)
    grid = np.stack([col, row], axis=0).astype(np.float32)  # (2,H,W)
    stride = np.array([IMG_SIZE / H, IMG_SIZE / W], dtype=np.float32).reshape(2, 1, 1)

    d = dfl_tensor(position)  # (4,H,W)
    # top-left (x1,y1)
    xy1 = (grid + 0.5 - d[:2]) * stride  # (2,H,W)
    # bottom-right (x2,y2)
    xy2 = (grid + 0.5 + d[2:]) * stride  # (2,H,W)
    xyxy = np.concatenate([xy1, xy2], axis=0)  # (4,H,W)
    return xyxy.reshape(4, -1).T  # (H*W,4)


def nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = NMS_THRESH
) -> np.ndarray:
    """
    纯 NumPy 实现的 NMS，返回保留的索引。
    boxes: (N,4) [x1,y1,x2,y2]
    scores: (N,)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 与剩余 box 计算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留低 IoU
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def yolov8_postprocess(
    input_data: list, original_shape: tuple, ratio: tuple, dw: float, dh: float
):
    """
    完整后处理：DFL 解码 + 阈值筛选 + NMS + 缩放回原图。
    input_data: 长度 DEFAULT_BRANCH*2，偶数位 feats, 奇数位 class_conf，batch=1。
    original_shape: (H0, W0)
    ratio: (rw, rh)
    dw, dh: padding
    """
    pair = len(input_data) // DEFAULT_BRANCH
    all_boxes, all_scores, all_cls = [], [], []

    for i in range(DEFAULT_BRANCH):
        pos = input_data[i * pair]  # (1, C, H, W)
        cls_conf = input_data[i * pair + 1]  # (1, num_classes, H, W)

        # squeeze batch
        pos = pos[0]
        cls_conf = cls_conf[0]
        num_classes = cls_conf.shape[0]
        H, W = pos.shape[1], pos.shape[2]

        # DFL 解码并平铺
        bb = box_process(pos)  # (H*W, 4)
        # 类别预测平铺
        cls_flat = cls_conf.transpose(1, 2, 0).reshape(
            -1, num_classes
        )  # (H*W, num_classes)

        # 置信分 = class_score * objectness(=1)
        cls_max = cls_flat.max(axis=1)
        cls_id = cls_flat.argmax(axis=1)
        scores = cls_max
        mask = scores >= OBJ_THRESH
        if not mask.any():
            continue

        all_boxes.append(bb[mask])
        all_scores.append(scores[mask])
        all_cls.append(cls_id[mask])

    if not all_boxes:
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,))

    # 拼接所有 branch
    boxes = np.vstack(all_boxes)  # (N,4)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_cls)

    # per-class NMS
    final_boxes, final_scores, final_cls = [], [], []
    for c in np.unique(classes):
        inds = np.where(classes == c)[0]
        b = boxes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        final_boxes.append(b[keep])
        final_scores.append(s[keep])
        final_cls.append(np.full(len(keep), c, dtype=int))

    boxes = np.concatenate(final_boxes)
    scores = np.concatenate(final_scores)
    classes = np.concatenate(final_cls)

    # 缩放回原图
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio[0]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio[1]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])

    return boxes, classes, scores


def draw_bbox(image, boxes, classes, scores, save_path, count):

    if not isinstance(classes, list) and not isinstance(classes, np.ndarray):
        l, t, r, b = (int(x) for x in boxes)

        cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{classes} id: {scores:.2f}",
            (l, t - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    else:
        for bbox, cl_name, score in zip(boxes, classes, scores):
            l, t, r, b = (int(x) for x in bbox)

            cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(
                image,
                f"{cl_name} id: {score:.2f}",
                (l, t - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    cv2.imwrite(os.path.join(save_path, f"output_{count}.jpg"), image)
    print(f"保存检测结果到 >> {os.path.join(save_path, f"output_{count}.jpg")}")
