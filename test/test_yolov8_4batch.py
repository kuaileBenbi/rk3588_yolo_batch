import os
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from utils import letterbox, yolov8_postprocess, draw_bbox

IMG_SIZE = 640

core_map = {
    "NPU_CORE_0": RKNNLite.NPU_CORE_0,
    "NPU_CORE_1": RKNNLite.NPU_CORE_1,
    "NPU_CORE_2": RKNNLite.NPU_CORE_2,
    "NPU_CORE_012": RKNNLite.NPU_CORE_0_1_2,
}


def draw(raw_batch, ltrb_boxes_batch, classes_id_batch, scores_batch, save_path):
    os.makedirs(save_path, exist_ok=True)
    count = 0
    for raw, ltrb_boxes, classes_id, scores in zip(
        raw_batch, ltrb_boxes_batch, classes_id_batch, scores_batch
    ):
        draw_bbox(raw, ltrb_boxes, classes_id, scores, save_path, count)
        count += 1


def init_rknn(det_model, core_mask):
    print("正在初始化 NPU")
    rknn_lite = RKNNLite()

    if rknn_lite.load_rknn(det_model) != 0:
        raise RuntimeError(f"加载 RKNN 模型失败: {det_model}")

    print("加载模型成功！")

    if rknn_lite.init_runtime(core_mask=core_map[core_mask]) != 0:
        raise RuntimeError(f"初始化 RKNN 运行时失败（core {core_mask}）")

    print(f"设置NPU 核心为: {core_mask}")

    return rknn_lite


def postprogress(infer_outputs, raw_shape, ratio, dw, dh):
    img1_outputs = []
    img2_outputs = []
    img3_outputs = []
    img4_outputs = []

    ltrb_boxes_batch = []
    classes_id_batch = []
    scores_batch = []

    for _, header in enumerate(infer_outputs):
        img1_outputs.append(np.expand_dims(header[0], axis=0))
        img2_outputs.append(np.expand_dims(header[1], axis=0))
        img3_outputs.append(np.expand_dims(header[2], axis=0))
        img4_outputs.append(np.expand_dims(header[3], axis=0))

    for outputs in [img1_outputs, img2_outputs, img3_outputs, img4_outputs]:
        ltrb_boxes, classes_id, scores = yolov8_postprocess(
            outputs, raw_shape, ratio, dw, dh
        )
        ltrb_boxes_batch.append(ltrb_boxes)
        classes_id_batch.append(classes_id)
        scores_batch.append(scores)

    return ltrb_boxes_batch, classes_id_batch, scores_batch


def preprocess(raw_batch_path):

    imgs = os.listdir(raw_batch_path)

    raw_batch = []
    raws = []

    for imgname in imgs:
        print(f"正在准备图像：{imgname}")
        raw_img_path = os.path.join(raw_batch_path, imgname)
        raw_img = cv2.imread(raw_img_path)
        raws.append(raw_img)
        print(f"图像大小为：{raw_img.shape}")
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        pre_img, ratio, (dw, dh) = letterbox(raw_img, new_shape=(IMG_SIZE, IMG_SIZE))
        pre_img = np.expand_dims(pre_img, 0)
        raw_batch.append(pre_img)

    img_batch = np.concatenate(raw_batch, axis=0)

    print("已经读取所有原始图像！")

    return raws, img_batch, ratio, (dw, dh)


def inference_and_postprocess(img_batch, rknn_lite, raw_img_shape, ratio, dw, dh):

    t1 = time.perf_counter_ns()
    outputs = rknn_lite.inference(inputs=[img_batch], data_format=["nhwc"])
    t2 = time.perf_counter_ns()

    print(f"[RKNN] 推理耗时 : {(t2-t1)/1e6:.1f} ms")

    t1 = time.perf_counter_ns()
    ltrb_boxes, classes_id, scores = postprogress(outputs, raw_img_shape, ratio, dw, dh)
    t2 = time.perf_counter_ns()
    print(f"[RKNN] 后处理耗时 : {(t2-t1)/1e6:.1f} ms")

    return ltrb_boxes, classes_id, scores


def main(rknn_model_path, core_mask, raw_batch_path, save_path):

    raw_batch, img_batch, ratio, (dw, dh) = preprocess(raw_batch_path)

    print(">> 准备推理...")

    rknn_lite = init_rknn(rknn_model_path, core_mask)

    print(">> 开始推理...")
    print(raw_batch[0].shape)
    ltrb_boxes, classes_id, scores = inference_and_postprocess(
        img_batch, rknn_lite, raw_batch[0].shape, ratio, dw, dh
    )

    print(">> 完成推理")

    draw(raw_batch, ltrb_boxes, classes_id, scores, save_path)

    print(f"本次所有推理工作全部结束！结果已保存至：{save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        rknn_model_path = sys.argv[1]
        core_mask = sys.argv[2]
        raw_batch_path = sys.argv[3]
        save_path = sys.argv[4]
    else:
        rknn_model_path = "/home/firefly/Desktop/tracker-methods/creative/yolo_batch/test/rknnModel/yolov8n_4batch.rknn"
        core_mask = "NPU_CORE_012"
        raw_batch_path = (
            "/home/firefly/Desktop/tracker-methods/creative/yolo_batch/test/data"
        )
        save_path = "debug_output"

    main(rknn_model_path, core_mask, raw_batch_path, save_path)
