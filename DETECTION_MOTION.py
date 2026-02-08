import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETTINGS & CALIBRATION ---
K = np.array([[1855, 0, 574], [0, 1850, 965], [0, 0, 1]], dtype=float)
inv_K = np.linalg.inv(K)


def auto_detect_features(frame):
    H, W, _ = frame.shape
    output = frame.copy()  # Create a copy for drawing

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 1. LIGHTS (E, A, B, F) ---
    _, thresh_lights = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh_lights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get top 2 lights
    light_cnts = sorted([c for c in cnts if cv2.contourArea(c) > 40], key=cv2.contourArea, reverse=True)[:2]
    if len(light_cnts) < 2: return None, frame

    # Sort Left and Right
    light_cnts = sorted(light_cnts, key=lambda c: cv2.boundingRect(c)[0])
    cntL, cntR = light_cnts[0], light_cnts[1]

    # Extract Points
    E = cntL[np.argmin(cntL[:, 0, 1])][0]  # Top L
    A = cntL[np.argmax(cntL[:, 0, 0])][0]  # Inner L
    F = cntR[np.argmin(cntR[:, 0, 1])][0]  # Top R
    B = cntR[np.argmin(cntR[:, 0, 0])][0]  # Inner R

    # --- 2. PLATE (C, D) ---
    _, thresh_plate = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    cnts_p, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    C, D = None, None
    best_match_val = 0

    for cnt in cnts_p:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # Heuristic: Plate is between A and B and below E
        if 2.2 < aspect_ratio < 6.5 and area > 80:
            if x > A[0] - 60 and (x + w) < B[0] + 60 and y > E[1]:
                if area > best_match_val:
                    best_match_val = area
                    # Simple bounding box top edges are robust enough here
                    C = np.array([x, y])
                    D = np.array([x + w, y])

    if C is None: return None, frame  # Require plate for geometry

    pts = {'E': E, 'A': A, 'B': B, 'F': F, 'C': C, 'D': D}

    # --- 3. DRAWING (VISUALIZATION FIX) ---
    # Draw Model Lines (Yellow: E->A->B->F)
    cv2.line(output, tuple(E), tuple(A), (0, 255, 255), 2)  # Yellow
    cv2.line(output, tuple(A), tuple(B), (0, 255, 255), 2)
    cv2.line(output, tuple(B), tuple(F), (0, 255, 255), 2)

    # Draw Plate Line (Blue: C->D)
    cv2.line(output, tuple(C), tuple(D), (255, 0, 0), 2)  # Blue

    # Draw Points (Green Circles + Labels)
    for label, pt in pts.items():
        cv2.circle(output, tuple(pt), 5, (0, 255, 0), -1)
        cv2.putText(output, label, (pt[0] - 5, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return pts, output


def get_robust_horizontal_direction(pts):
    """Calculates dx, forcing Infinity if lines are nearly parallel."""

    def h(p):
        return np.array([p[0], p[1], 1.0])

    # Line 1: AB
    v_AB = pts['B'] - pts['A']
    angle_AB = np.arctan2(v_AB[1], v_AB[0])

    # Line 2: CD
    v_CD = pts['D'] - pts['C']
    angle_CD = np.arctan2(v_CD[1], v_CD[0])

    # Angle Difference
    angle_diff = abs(np.degrees(angle_AB - angle_CD))

    # If lines are parallel (diff < 7 degrees), V_x is at infinity
    if angle_diff < 7.0:
        # Average angle
        avg_ang = (angle_AB + angle_CD) / 2.0
        # V_x at infinity = direction vector [cos, sin, 0]
        vx = np.array([np.cos(avg_ang), np.sin(avg_ang), 0.0])
    else:
        # Standard intersection
        line_ab = np.cross(h(pts['A']), h(pts['B']))
        line_cd = np.cross(h(pts['C']), h(pts['D']))
        vx = np.cross(line_ab, line_cd)
        if abs(vx[2]) > 1e-5: vx = vx / vx[2]

    # dx = K_inv * vx
    dx = inv_K @ vx
    dx /= np.linalg.norm(dx)
    if dx[0] < 0: dx = -dx  # Point right
    return dx


def check_motion_orthogonality(pts1, pts2):
    """Checks angle between Horizontal (dx) and Forward (dy)."""

    def h(p):
        return np.array([p[0], p[1], 1.0])

    # 1. Horizontal (dx)
    dx = get_robust_horizontal_direction(pts1)

    # 2. Forward (dy) - Use E and F (most stable points)
    # Line connecting E in frame 1 to E in frame 2
    line_E1E2 = np.cross(h(pts1['E']), h(pts2['E']))
    line_F1F2 = np.cross(h(pts1['F']), h(pts2['F']))

    vy = np.cross(line_E1E2, line_F1F2)

    # Normalize Vy
    if abs(vy[2]) < 1e-9:
        dy = np.array([0, 0, 1])  # Pure forward Z
    else:
        vy = vy / vy[2]
        dy = inv_K @ vy
        dy /= np.linalg.norm(dy)

    # 3. Angle
    dot_prod = np.clip(np.dot(dx, dy), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_prod))

    return angle_deg


# --- EXECUTION ---
video_path = 'MovingForward.mp4'
frame_indices = [5, 120]

cap = cv2.VideoCapture(video_path)
data = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        pts, frame_drawn = auto_detect_features(frame)
        if pts:
            data.append({'frame': frame_drawn, 'pts': pts})
cap.release()

if len(data) == 2:
    print("\n--- Geometric Analysis ---")
    angle = check_motion_orthogonality(data[0]['pts'], data[1]['pts'])

    print(f"Calculated Angle: {angle:.2f} degrees")
    print(f"Deviation from 90: {abs(90 - angle):.2f} degrees")

    if abs(angle - 90) < 15.0:
        print("Conclusion: ORTHOGONAL (Forward Translation)")
    else:
        print("Conclusion: STEERING")

    # Show images
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for i, d in enumerate(data):
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(d['frame'], cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Frame {frame_indices[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not detect features in both frames.")