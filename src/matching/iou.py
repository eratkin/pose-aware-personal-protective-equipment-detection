def area(box):
    if box is None:
        return 0.0

    x1, y1, x2, y2 = box

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_box(box_a, box_b):
    if box_a is None or box_b is None:
        return None

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def intersection_area(box_a, box_b):
    intersection = intersection_box(box_a, box_b)

    if intersection is None:
        return 0.0

    return area(intersection)


def intersection_over_union(box_a, box_b):
    intersection = intersection_area(box_a, box_b)

    if intersection <= 0:
        return 0.0

    union = area(box_a) + area(box_b) - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def intersection_over_detection(detection_box, region_box):
    """
    How much of the detected object lies inside the expected region?

    Useful because a hard hat can be much smaller than the generated
    head region, producing a low IoU even when it is correctly located.
    """

    detection_area = area(detection_box)

    if detection_area <= 0:
        return 0.0

    return (
        intersection_area(detection_box, region_box)
        / detection_area
    )


def center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def point_in_box(point, box):
    if point is None or box is None:
        return False

    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and y1 <= py <= y2
    )


def center_in_box(detection_box, region_box):
    return point_in_box(
        center(detection_box),
        region_box,
    )


def box_center_distance(box_a, box_b):
    ax, ay = center(box_a)
    bx, by = center(box_b)

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5