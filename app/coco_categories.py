"""COCO 80-class contiguous label ids (0-79) and their 12 supercategories."""

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

_SUPERCATEGORY_RANGES = [
    ("person", 0, 1),
    ("vehicle", 1, 9),
    ("outdoor", 9, 14),
    ("animal", 14, 24),
    ("accessory", 24, 29),
    ("sports", 29, 39),
    ("kitchen", 39, 46),
    ("food", 46, 56),
    ("furniture", 56, 62),
    ("electronic", 62, 68),
    ("appliance", 68, 73),
    ("indoor", 73, 80),
]

SUPERCATEGORIES = [name for name, _, _ in _SUPERCATEGORY_RANGES]

LABEL_TO_SUPERCATEGORY = {
    label_id: name
    for name, start, end in _SUPERCATEGORY_RANGES
    for label_id in range(start, end)
}
