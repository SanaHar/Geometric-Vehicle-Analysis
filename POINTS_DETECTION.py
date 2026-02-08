import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. SETTINGS & CALIBRATION ---
K = np.array([[1855, 0, 574], [0, 1850, 965], [0, 0, 1]], dtype=float)
inv_K = np.linalg.inv(K)


def auto_detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 1. LIGHTS DETECTION (High Intensity) ---
    _, thresh_lights = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh_lights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    light_cnts = sorted([c for c in cnts if cv2.contourArea(c) > 40], key=cv2.contourArea, reverse=True)[:2]

    if len(light_cnts) < 2: return None, frame
    light_cnts = sorted(light_cnts, key=lambda c: cv2.boundingRect(c)[0])
    cntL, cntR = light_cnts[0], light_cnts[1]

    E = cntL[np.argmin(cntL[:, 0, 1])][0]
    A = cntL[np.argmax(cntL[:, 0, 0])][0]
    F = cntR[np.argmin(cntR[:, 0, 1])][0]
    B = cntR[np.argmin(cntR[:, 0, 0])][0]

    # --- 2. ROBUST PLATE DETECTION (Lower threshold + Adaptive filters) ---
    # We lower the threshold to 120 and area to 100 to catch distant plates
    _, thresh_plate = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    cnts_p, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    C, D = None, None
    best_match_val = 0

    for cnt in cnts_p:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # Heuristics for European Plate
        if 2.2 < aspect_ratio < 6.5 and area > 80:
            # Must be horizontally between A and B (with 60px tolerance for perspective)
            if x > A[0] - 60 and (x + w) < B[0] + 60:
                # Must be vertically below the lights
                if y > E[1]:
                    # Pick the largest valid candidate (usually the reflective plate)
                    if area > best_match_val:
                        best_match_val = area
                        C = np.array([x, y])
                        D = np.array([x + w, y])

    pts = {'E': E, 'A': A, 'B': B, 'F': F, 'C': C, 'D': D}
    return pts, frame


def get_plane_orientation(pts):
    def h(p):
        return np.array([p[0], p[1], 1.0])

    line_ab = np.cross(h(pts['A']), h(pts['B']))

    # Use CD if available, otherwise fallback to EF (Slide 31 iterative logic)
    if pts['C'] is not None and pts['D'] is not None:
        line_ref = np.cross(h(pts['C']), h(pts['D']))
    else:
        line_ref = np.cross(h(pts['E']), h(pts['F']))

    vx = np.cross(line_ab, line_ref)
    vx = vx / vx[2] if abs(vx[2]) > 1e-9 else vx
    dx = inv_K @ vx
    dx /= np.linalg.norm(dx)
    if dx[0] < 0: dx = -dx
    return dx


# --- 2. EXECUTION ---
video_path = 'MovingForward.mp4'
filename = os.path.basename(video_path).lower()
frame_indices = [0, 60, 120] if "randomly" in filename else [0, 100]

cap = cv2.VideoCapture(video_path)
data = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        pts, _ = auto_detect_features(frame)
        if pts:
            data.append({'frame': frame, 'pts': pts, 'dir': get_plane_orientation(pts)})
cap.release()

# --- 3. VISUALIZATION ---
fig, axes = plt.subplots(1, len(data), figsize=(14, 6))
for i, entry in enumerate(data):
    ax = axes[i]
    ax.imshow(cv2.cvtColor(entry['frame'], cv2.COLOR_BGR2RGB))
    p = entry['pts']

    # Draw Yellow Model (E-A-B-F)
    model_line = np.array([p['E'], p['A'], p['B'], p['F']], np.int32)
    cv2.polylines(entry['frame'], [model_line], False, (0, 255, 255), 2)

    # Draw Blue Plate (C-D)
    if p['C'] is not None:
        cv2.line(entry['frame'], tuple(p['C']), tuple(p['D']), (255, 0, 0), 3)

    ax.imshow(cv2.cvtColor(entry['frame'], cv2.COLOR_BGR2RGB))
    for label, pt in p.items():
        if pt is not None:
            ax.scatter(pt[0], pt[1], s=30, c='lime')
            ax.text(pt[0], pt[1] - 15, label, color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=0.5))
    ax.set_title(f"Auto-Detection Frame {i + 1}")
    ax.axis('off')

plt.tight_layout()
plt.show()