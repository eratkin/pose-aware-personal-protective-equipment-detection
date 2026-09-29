def _clip(box, width, height):
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(width - 1, int(x2)), min(height - 1, int(y2)))


def expected_regions(person, image_shape, cfg):
    """Create expected helmet/vest regions from normalized semantic landmarks.

    person['points'] keys can include nose, left_ear, right_ear, left_shoulder,
    right_shoulder, left_hip, right_hip. Each value is (x, y, confidence).
    """
    h, w = image_shape[:2]
    p = person.get('points', {})
    r = cfg['regions']
    out = {}

    ls, rs = p.get('left_shoulder'), p.get('right_shoulder')
    nose, le, re = p.get('nose'), p.get('left_ear'), p.get('right_ear')
    lh, rh = p.get('left_hip'), p.get('right_hip')

    head_pts = [q for q in (nose, le, re) if q and q[2] > 0]
    if head_pts and ls and rs:
        shoulder_span = max(20.0, abs(rs[0] - ls[0]))
        cx = sum(q[0] for q in head_pts) / len(head_pts)
        cy = sum(q[1] for q in head_pts) / len(head_pts)
        hw = shoulder_span * 0.5 * r['head_width_scale']
        hh = shoulder_span * 0.45 * r['head_height_scale']
        out['hard_hat'] = _clip((cx - hw, cy - hh * 1.15, cx + hw, cy + hh * 0.35), w, h)

    if ls and rs:
        top_y = min(ls[1], rs[1])
        shoulder_span = max(20.0, abs(rs[0] - ls[0]))
        cx = (ls[0] + rs[0]) / 2
        if lh and rh:
            bottom_y = max(lh[1], rh[1])
        else:
            bottom_y = top_y + shoulder_span * 1.6
        half_w = shoulder_span * 0.5 * r['torso_width_scale']
        height = max(20.0, bottom_y - top_y) * r['torso_height_scale']
        out['safety_vest'] = _clip((cx - half_w, top_y, cx + half_w, top_y + height), w, h)

    return out
