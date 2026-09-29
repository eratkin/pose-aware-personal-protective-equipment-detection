from .iou import (
    intersection_over_union,
    center,
    point_in_box,
)


def pose_reference_point(pose):
    """
    Get a useful central point from a pose.

    Shoulders are preferred because they tend to be more stable than
    extremities.
    """

    points = pose.get("points", {})

    candidates = []

    for name in [
        "left_shoulder",
        "right_shoulder",
        "nose",
        "left_hip",
        "right_hip",
    ]:
        point = points.get(name)

        if point is None:
            continue

        x, y, confidence = point

        if confidence >= 0.20:
            candidates.append((x, y))

    if not candidates:
        return None

    x = sum(p[0] for p in candidates) / len(candidates)
    y = sum(p[1] for p in candidates) / len(candidates)

    return (x, y)


def association_score(person_box, pose):
    """
    Score how likely a pose belongs to a Roboflow-detected person.

    Roboflow person detection remains authoritative.
    """

    pose_box = pose.get("box")

    score = 0.0

    if pose_box is not None:
        score += intersection_over_union(
            person_box,
            pose_box,
        )

    reference = pose_reference_point(pose)

    if reference is not None and point_in_box(
        reference,
        person_box,
    ):
        # Strong bonus if the central pose landmarks actually fall
        # inside the Roboflow person box.
        score += 1.0

    return score


def associate_poses_to_people(
    person_detections,
    poses,
    minimum_score=0.05,
):
    """
    Match pose results to Roboflow person detections.

    Returns one record for EVERY Roboflow person.

    If no pose can be associated, pose=None. The person still exists
    because Roboflow detected them.
    """

    assignments = []

    unused_pose_indices = set(
        range(len(poses))
    )

    # Higher-confidence Roboflow person detections are processed first.
    sorted_people = sorted(
        person_detections,
        key=lambda d: d.get("confidence", 0.0),
        reverse=True,
    )

    for person_index, person in enumerate(
        sorted_people,
        start=1,
    ):
        best_pose_index = None
        best_score = 0.0

        for pose_index in unused_pose_indices:
            pose = poses[pose_index]

            score = association_score(
                person["box"],
                pose,
            )

            if score > best_score:
                best_score = score
                best_pose_index = pose_index

        matched_pose = None

        if (
            best_pose_index is not None
            and best_score >= minimum_score
        ):
            matched_pose = poses[best_pose_index]
            unused_pose_indices.remove(best_pose_index)

        assignments.append(
            {
                "person_id": person_index,
                "person_detection": person,
                "pose": matched_pose,
                "pose_match_score": round(
                    best_score,
                    4,
                ),
            }
        )

    return assignments