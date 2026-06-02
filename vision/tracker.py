import numpy as np


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


class Track:
    def __init__(self, track_id, box, embedding, det_score=1.0):
        self.id = track_id
        self.box = box
        self.embedding = embedding
        self.det_score = det_score   # InsightFace detection confidence
        self.name = "..."
        self.confidence = 0.0
        self.lost_frames = 0
        # Force recognition on first appearance
        self.frames_since_recognition = 9999


class FaceTracker:
    """
    IoU-based tracker. Each detected face box is matched to an existing track
    by IoU overlap. Unmatched detections become new tracks. Tracks not matched
    for max_lost consecutive frames are dropped.

    Recognition is requested only when a track is new or has not been
    re-confirmed for rerecognize_every frames — so the heavy embedding lookup
    runs once per identity rather than 30 times per second.
    """

    def __init__(self, iou_threshold=0.35, max_lost=20, rerecognize_every=90):
        self._tracks = {}        # track_id → Track
        self._next_id = 1
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.rerecognize_every = rerecognize_every

    def update(self, detections):
        """
        detections: list of (box_ndarray, embedding_ndarray, det_score)
        Returns: list of Track objects that are currently visible.
        """
        matched_ids = set()
        active = []

        for box, emb, det_score in detections:
            best_score = self.iou_threshold
            best_track = None

            for t in self._tracks.values():
                if t.id in matched_ids:
                    continue
                score = _iou(box, t.box)
                if score > best_score:
                    best_score = score
                    best_track = t

            if best_track is not None:
                best_track.box = box
                best_track.embedding = emb
                best_track.det_score = det_score
                best_track.lost_frames = 0
                best_track.frames_since_recognition += 1
                matched_ids.add(best_track.id)
                active.append(best_track)
            else:
                t = Track(self._next_id, box, emb, det_score)
                self._next_id += 1
                self._tracks[t.id] = t
                active.append(t)

        # Age unmatched tracks and prune dead ones
        for t in list(self._tracks.values()):
            if t.id not in matched_ids:
                t.lost_frames += 1
        self._tracks = {
            k: v for k, v in self._tracks.items()
            if v.lost_frames <= self.max_lost
        }

        return active

    def confirm(self, track, name, confidence):
        track.name = name
        track.confidence = confidence
        track.frames_since_recognition = 0

    def needs_recognition(self, track):
        return track.frames_since_recognition >= self.rerecognize_every
