import logging
import re
import json
import math

from datasets import Dataset

eval_logger = logging.getLogger("lmms-eval")

COCO_REC_METRICS = ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.7", "ACC@0.9", "Center_ACC"]


def refcoco_bbox_rec_preprocess_dataset(dataset: Dataset):
    # PIL image stored in dataset['image']
    # add `image_width` and `image_height` to the dataset
    dataset = dataset.map(lambda x: {"image_width": x["image"].width, "image_height": x["image"].height})

    # Original bbox format (top x, top y, width, height)
    # Convert to (top-left x, top-left y, bottom-right x, bottom-right y)
    # Normalize the bounding box coordinates to be between 0 and 1
    # using the image width and height
    dataset = dataset.map(lambda x: {"bbox": [x["bbox"][0] / x["image_width"], x["bbox"][1] / x["image_height"],
                                              (x["bbox"][0] + x["bbox"][2]) / x["image_width"],
                                              (x["bbox"][1] + x["bbox"][3]) / x["image_height"]]})

    # currently, the dataset has `answer` as a list of strings
    # each answer should be its own row
    # we will explode the dataset to have one row per answer
    # duplicate the other columns
    def explode_answers(example):
        answers = example.pop("answer")
        return [{"answer": answer, **example} for answer in answers]

    # Apply the function to each element, collecting the results
    exploded_rows = []
    for example in dataset:
        exploded_rows.extend(explode_answers(example))

    # Create a new dataset from the exploded rows
    new_dataset = Dataset.from_list(exploded_rows)
    print(f"Exploded dataset from {len(dataset)} to {len(new_dataset)} rows")

    return new_dataset


def refcoco_bbox_rec_doc_to_visual(doc):
    # Image is presented as is
    image = doc["image"].convert("RGB")
    return [image.convert("RGB")]


def refcoco_bbox_rec_doc_to_text(doc):
    assert isinstance(doc["answer"], str), "Answer must be a string"
    return f'Please provide the bounding box coordinate of the region this sentence describes: {doc["answer"]}'


# Locate {doc["answer"]} in the image and return the location in the form of coordinates in the format {{"bbox_2d": [x1, y1, x2, y2]}}
# Please provide the bounding box coordinate of the region this sentence describes: {doc["answer"]}

def parse_float_sequence_within(input_str):
    # 处理 JSON 代码块格式
    if "```json" in input_str or "```" in input_str:
        try:
            if "```json" in input_str:
                json_str = input_str.split("```json")[1].split("```")[0].strip()
            else:
                json_str = input_str.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)

            # 4 个数字的列表 [x1, y1, x2, y2]
            if isinstance(data, list) and len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
                return [float(x) for x in data]

            # 字典列表 [{"bbox_2d": [...]}]
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                if isinstance(item, dict) and "bbox_2d" in item:  # ✅ 先检查是否为 dict
                    bbox = item["bbox_2d"]
                    if len(bbox) == 4:
                        return [float(x) for x in bbox]

            # 单个字典 {"bbox_2d": [...]}
            elif isinstance(data, dict) and "bbox_2d" in data:
                bbox = data["bbox_2d"]
                if len(bbox) == 4:
                    return [float(x) for x in bbox]

        except (json.JSONDecodeError, ValueError, IndexError, KeyError):
            pass

    # 取 bbox_2d 数组
    bbox_2d_pattern = r'"bbox_2d"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
    match = re.search(bbox_2d_pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # 元组格式 (x1,y1),(x2,y2)
    tuple_pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
    match = re.search(tuple_pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # 标准数组格式 [x1, y1, x2, y2]
    array_pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(array_pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]

    # 提取前4个数字（兜底）
    numbers = re.findall(r'\d+(?:\.\d+)?', input_str)
    if len(numbers) >= 4:
        coords = [float(numbers[i]) for i in range(4)]
        if all(0 <= c <= 10000 for c in coords):
            return coords

    if input_str.strip():
        eval_logger.debug(f"Failed to parse bbox from: {input_str[:200]}")

    return [0, 0, 0, 0]


def refcoco_bbox_rec_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""
    pred_bbox = parse_float_sequence_within(pred)
    eval_logger.debug(f"Raw prediction: {pred_bbox}")

    if pred_bbox != [0, 0, 0, 0]:
        if max(pred_bbox) > 2:
            orig_w = doc["image_width"]
            orig_h = doc["image_height"]

            pred_bbox = [
                pred_bbox[0] / orig_w,
                pred_bbox[1] / orig_h,
                pred_bbox[2] / orig_w,
                pred_bbox[3] / orig_h,
            ]
            eval_logger.debug(f"Normalized to [0,1]: {pred_bbox}")

    pred_bbox = [max(0.0, min(1.0, x)) for x in pred_bbox]

    eval_logger.debug(f"Final normalized bbox: {pred_bbox}")
    eval_logger.debug(f"GT bbox: {doc['bbox']}")

    ann_id = doc["question_id"]
    data_dict = {
        "answer": doc["answer"],
        "pred": pred_bbox,
        "ann_id": ann_id,
        "bbox": doc["bbox"]
    }
    return {f"refcoco_{metric}": data_dict for metric in COCO_REC_METRICS}


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Compute the Intersection over Union
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def compute_accuracy(box1, box2, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    iou = compute_iou(box1, box2)
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def refcoco_bbox_rec_aggregation_result(results, metric):
    """
    Aggregate the results of the RefCOCO evaluation task using the specified metric.

    Args:
    - results (list of dict): List of result dictionaries.
    - metric (str): Metric to use for aggregation.

    Returns:
    - dict: Dictionary containing the aggregated results for the specified metric.
    """
    scorers = {
        "IoU": compute_iou,
        "ACC@0.1": lambda x, y: compute_accuracy(x, y, 0.1),
        "ACC@0.3": lambda x, y: compute_accuracy(x, y, 0.3),
        "ACC@0.5": lambda x, y: compute_accuracy(x, y, 0.5),
        "ACC@0.7": lambda x, y: compute_accuracy(x, y, 0.7),
        "ACC@0.9": lambda x, y: compute_accuracy(x, y, 0.9),
        "Center_ACC": compute_center_accuracy,
    }
    results_dict = {metric: []}
    for result in results:
        # Extract the ground truth and predicted bounding boxes
        gt_bbox = result["bbox"]
        pred_bbox = result["pred"]
        # Compute the specified metric between the ground truth and predicted bounding boxes
        score = scorers[metric](gt_bbox, pred_bbox)
        results_dict[metric].append(score)
    results_dict[metric] = sum(results_dict[metric]) / len(results_dict[metric])
    print(f"Aggregated {metric} score: {results_dict[metric]}")
    return results_dict[metric]


def refcoco_bbox_rec_iou(results):
    return refcoco_bbox_rec_aggregation_result(results, "IoU")


def refcoco_bbox_rec_acc01(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.1")


def refcoco_bbox_rec_acc03(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.3")


def refcoco_bbox_rec_acc05(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.5")


def refcoco_bbox_rec_acc07(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.7")


def refcoco_bbox_rec_acc09(results):
    return refcoco_bbox_rec_aggregation_result(results, "ACC@0.9")


def refcoco_bbox_rec_center_acc(results):
    return refcoco_bbox_rec_aggregation_result(results, "Center_ACC")