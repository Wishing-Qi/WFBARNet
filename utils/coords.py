import numpy as np

def scale_coords(coords, from_shape, to_shape, is_keypoints=False):
    """
    将坐标从 from_shape 缩放到 to_shape。
    coords: [x, y] 或 [x1, y1, x2, y2] 或 keypoints [N, 3]
    from_shape: [h, w]
    to_shape: [h, w]
    """
    fh, fw = from_shape[:2]
    th, tw = to_shape[:2]
    
    sh = th / fh
    sw = tw / fw

    if is_keypoints:
        # keypoints shape usually [N, 3] (x, y, conf)
        new_coords = coords.copy()
        new_coords[:, 0] *= sw
        new_coords[:, 1] *= sh
        return new_coords
    
    if len(coords) == 2: # [x, y]
        return [coords[0] * sw, coords[1] * sh]
    elif len(coords) == 4: # [x1, y1, x2, y2]
        return [coords[0] * sw, coords[1] * sh, coords[2] * sw, coords[3] * sh]
    
    return coords

def normalize_to_pixel(point_or_bbox, model_input_size, original_size):
    """
    便捷包装：模型推断坐标 -> 原始视频坐标
    model_input_size: (w, h)
    original_size: (w, h)
    """
    mw, mh = model_input_size
    ow, oh = original_size
    return scale_coords(point_or_bbox, (mh, mw), (oh, ow))
