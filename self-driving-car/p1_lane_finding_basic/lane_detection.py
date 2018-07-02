import numpy as np
import cv2
from line import Line


def smoothen_over_time(lane_lines):
    """
    smooth the lane line inference over a window of frames
    :param lane_lines:
    :return:
    """
    avg_line_lt = np.zeros(shape=(len(lane_lines), 4))
    avg_line_rt = np.zeros(shape=(len(lane_lines), 4))
    for i, v in enumerate(lane_lines):
        avg_line_lt[i] += v[0].get_coords()
        avg_line_rt[i] += v[1].get_coords()
    return Line(*np.mean(avg_line_lt, axis=0)), Line(*np.mean(avg_line_rt, axis=0))


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)

def weighted_img(img, initial_img, alpha=0.8, beta=1., lam=0.):
    """
    Return initial_img * alpha + img * beta + lambda
    """
    img = np.uint8(img)
    if len(img.shape) == 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))
    return cv2.addWeighted(initial_img, alpha, img, beta, lam)


def compute_lane_from_candidates(line_candidates, img_shape):
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)
    return left_lane, right_lane


def get_lane_lines(color_image, solid_lines=True):
    img = cv2.resize(color_image, (960, 540))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)
    detected_lines = hough_lines_detection(img_edge,
                                           rho=2,
                                           theta=np.pi / 180,
                                           threshold=1,
                                           min_line_len=15,
                                           max_line_gap=5)
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]
    if solid_lines:
        candidate_lines = []
        for line in detected_lines:
            if 0.5 <= np.abs(line.slope) <= 2:
                candidate_lines.append(line)
        return compute_lane_from_candidates(candidate_lines, img_gray.shape)
    else:
        return detected_lines


def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    is_videoclip = len(frames) > 0
    img_h, img_w = frames[0].shape[0], frames[0].shape[1]

    lane_lines = []
    for t in range(0, len(frames)):
        inferred_lanes = get_lane_lines(color_image=frames[t], solid_lines=solid_lines)
        lane_lines.append(inferred_lanes)

    if temporal_smoothing and solid_lines:
        lane_lines = smoothen_over_time(lane_lines)
    else:
        lane_lines = lane_lines[0]

    img_masked = np.zeros(shape=(img_h, img_w))
    for lane in lane_lines:
        lane.draw(img_masked)

    img_color = frames[-1] if is_videoclip else frames[0]
    img_blend = weighted_img(img_masked, img_color, 0.8, 1, 0)
    return img_blend