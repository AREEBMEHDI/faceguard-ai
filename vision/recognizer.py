import os
import numpy as np
import cv2
import faiss
from insightface.app import FaceAnalysis

# CPU-only: avoids Metal conflict with PyTorch (YOLO) in the same process.
_PROVIDERS = ["CPUExecutionProvider"]

# det_size=(640,640) instead of (320,320):
# RetinaFace's minimum detectable face scales with det_size.
# At 320: misses faces smaller than ~30px wide (person > ~1.5 m away on webcam).
# At 640: detects faces down to ~15px wide (person ~3-4 m away on webcam).
# Slower per-frame, but necessary for any CCTV / wide-angle use case.
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=_PROVIDERS,
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# Per-photo embeddings (not one mean per person).
# Stored in FAISS; index position → person name via _labels.
_labels = []        # one entry per stored embedding
faiss_index = None


def load_known_faces(base_path="known_faces"):
    """
    Build the FAISS index from known_faces/<person>/*.jpg.

    Each photo becomes its own embedding plus a horizontally-flipped copy
    (simulates the face turned ~30° the other way, helping with side-profile
    and off-axis detections).  Majority vote at query time handles the
    resulting duplicate labels gracefully.
    """
    global faiss_index, _labels

    all_embeddings = []
    all_labels = []

    for person_name in sorted(os.listdir(base_path)):
        person_folder = os.path.join(base_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        count = 0
        for fname in os.listdir(person_folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(os.path.join(person_folder, fname))
            if img is None:
                continue

            for candidate in [img, cv2.flip(img, 1)]:   # original + mirror
                faces = app.get(candidate)
                if not faces:
                    continue
                emb = faces[0].embedding.astype("float32")
                all_embeddings.append(emb)
                all_labels.append(person_name)
                count += 1

        print(f"  {person_name}: {count} embeddings (from {count//2} photos × 2)")

    if not all_embeddings:
        print("⚠️  No face embeddings found in known_faces/")
        return

    matrix = np.array(all_embeddings, dtype="float32")
    faiss.normalize_L2(matrix)

    _labels = all_labels
    faiss_index = faiss.IndexFlatIP(matrix.shape[1])
    faiss_index.add(matrix)
    print(f"✅ FAISS index: {len(_labels)} embeddings for {len(set(_labels))} identities")


def detect_faces(frame):
    """
    Returns list of (box_int_array, normalized_embedding, det_score).
    det_score is the detection confidence (0-1); low = small/blurry face.
    """
    results = []
    for face in app.get(frame):
        box = face.bbox.astype(int)
        emb = face.embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(emb)
        results.append((box, emb, float(face.det_score)))
    return results


def identify(embedding, det_score=1.0):
    """
    Majority vote across the top-k FAISS neighbours.

    Threshold is relaxed for low-confidence detections (small/far faces)
    so a partially-visible face at range still triggers a match.
    Far/side faces genuinely are less similar to a frontal portrait
    embedding, so the threshold scales with detection quality.
    """
    if faiss_index is None or not _labels:
        return "Unknown", 0.0

    k = min(7, faiss_index.ntotal)
    similarities, indices = faiss_index.search(embedding, k=k)

    # Adaptive threshold: loosen for small/far/occluded faces
    if det_score >= 0.80:
        threshold = 0.38        # high-quality close-up face
    elif det_score >= 0.60:
        threshold = 0.32        # medium quality (some distance / angle)
    else:
        threshold = 0.27        # low quality (far away or profile)

    # Collect votes from neighbours that pass the threshold
    votes: dict[str, float] = {}
    for sim, idx in zip(similarities[0], indices[0]):
        if sim < threshold:
            continue
        name = _labels[idx]
        # Weight votes by similarity so closer matches count more
        votes[name] = votes.get(name, 0.0) + float(sim)

    if not votes:
        return "Unknown", float(similarities[0][0])

    best_name = max(votes, key=votes.__getitem__)
    best_sim = float(similarities[0][0])   # highest single similarity
    return best_name, best_sim
