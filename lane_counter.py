"""
lane_counter.py - Lane-based classified vehicle counting + density (traffic survey module)

Step 1 (once per camera position): draw the lanes on the first frame
    python lane_counter.py --source sample.mp4 --setup

Step 2: run the survey
    python lane_counter.py --source sample.mp4
    python lane_counter.py --source sample.mp4 --interval 900 --save annotated.mp4

Output: counts.csv with per-lane, per-interval classified counts, flow and density.
Press q in the video window to stop early.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ---------------------------------------------------------------- settings
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO class ids
CLASS_NAMES = list(VEHICLES.values())
MIN_FRAMES_IN_LANE = 5       # a vehicle must stay this many frames in a lane to be counted
LOW_MAX, HIGH_MIN = 2, 5     # avg vehicles in zone: < LOW_MAX = Low, >= HIGH_MIN = High (tune these)
DISPLAY_MAX = (1280, 720)    # largest window size (width, height) shown on screen
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0)]


def open_source(src):
    return cv2.VideoCapture(int(src) if str(src).isdigit() else src)


# ---------------------------------------------------------------- lane setup
def fit_scale(w, h):
    """Scale factor so the frame fits on screen (never upscales)."""
    return min(DISPLAY_MAX[0] / w, DISPLAY_MAX[1] / h, 1.0)


def show_scaled(win, img):
    s = fit_scale(img.shape[1], img.shape[0])
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imshow(win, img)


def setup_lanes(source, out_path):
    cap = open_source(source)
    idx = 0

    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(i, 0))
        ok, fr = cap.read()
        return fr if ok else None

    frame = grab(idx)
    if frame is None:
        sys.exit("Could not read a frame from the source.")
    h, w = frame.shape[:2]
    scale = fit_scale(w, h)

    lanes, current = [], []
    win = "Lane setup"
    entering = None     # None = normal mode; a string = typing the lane length in the window

    def finish_lane(length):
        name = f"Lane {len(lanes) + 1}"
        lanes.append({"name": name, "points": [list(p) for p in current], "length_m": length})
        current.clear()
        print(f"  {name} saved ({len(lanes)} lane(s) so far).")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([int(x / scale), int(y / scale)])   # store in original-frame coordinates

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("\nLANE SETUP")
    print("  d / a = jump forward / back 30 frames (find a frame where the road is clearly visible)")
    print("  Left-click to place corners of one lane (3+ points, along the road surface).")
    print("  n = finish this lane (then type its real length in metres IN THE VIDEO WINDOW,")
    print("      Enter to confirm, Enter with nothing typed to skip)")
    print("  u = undo last point   s = save and quit   q = quit without saving\n")

    while True:
        view = frame.copy()
        for i, lane in enumerate(lanes):
            pts = np.array(lane["points"], np.int32)
            color = COLORS[i % len(COLORS)]
            cv2.polylines(view, [pts], True, color, 3)
            cv2.putText(view, lane["name"], tuple(int(v) for v in pts[0]), FONT, 1.2, color, 3)
        for p in current:
            cv2.circle(view, tuple(p), 6, (0, 0, 255), -1)
        if len(current) > 1:
            cv2.polylines(view, [np.array(current, np.int32)], False, (0, 0, 255), 3)
        cv2.putText(view, f"frame {idx}", (10, 40), FONT, 1.0, (255, 255, 255), 3)
        if entering is not None:
            cv2.rectangle(view, (0, 60), (view.shape[1], 130), (0, 0, 0), -1)
            cv2.putText(view, f"Lane {len(lanes) + 1} length in metres: {entering}_   (Enter = confirm, empty = skip)",
                        (10, 105), FONT, 1.0, (0, 255, 255), 2)
        show_scaled(win, view)
        key = cv2.waitKey(30) & 0xFF

        if entering is not None:                       # typing the lane length
            if key in (13, 10, 27):                    # Enter or Esc
                try:
                    length = float(entering) if entering else None
                except ValueError:
                    length = None
                finish_lane(length)
                entering = None
            elif key == 8:                             # Backspace
                entering = entering[:-1]
            elif key < 128 and chr(key) in "0123456789.":
                entering += chr(key)
            continue

        if key == ord("d") or key == ord("a"):
            new_idx = idx + 30 if key == ord("d") else idx - 30
            fr = grab(new_idx)
            if fr is not None:
                frame, idx = fr, max(new_idx, 0)
        elif key == ord("n") and len(current) >= 3:
            entering = ""
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("s"):
            if lanes:
                with open(out_path, "w") as f:
                    json.dump({"lanes": lanes}, f, indent=2)
                print(f"Saved {len(lanes)} lane(s) to {out_path}")
            else:
                print("No lanes defined, nothing saved.")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break
    cap.release()
    cv2.destroyAllWindows()


def point_lane(polys, pt):
    for i, poly in enumerate(polys):
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return i
    return None


# ---------------------------------------------------------------- main run
def run(args):
    try:
        with open(args.lanes) as f:
            lanes = json.load(f)["lanes"]
    except FileNotFoundError:
        sys.exit(f"{args.lanes} not found. Run with --setup first to define the lanes.")

    polys = [np.array(l["points"], np.int32) for l in lanes]
    n = len(lanes)

    model = YOLO(args.model)
    tracker = DeepSort(max_age=30, n_init=3)
    cap = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval_frames = max(1, int(args.interval * fps))
    vid_writer = None

    counted = {}                          # track id -> lane index (each vehicle counted once)
    frames_in_lane = defaultdict(int)     # (track id, lane) -> frames spent in lane
    class_votes = defaultdict(Counter)    # track id -> votes per class (smooths flicker)
    totals = [Counter() for _ in range(n)]
    interval_counts = [Counter() for _ in range(n)]
    occ_sum = [0] * n
    occ_peak = [0] * n
    interval_frames_seen = 0
    interval_start = 0.0

    csv_file = open(args.out, "w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["interval_start_s", "interval_end_s", "lane", *CLASS_NAMES, "total",
                    "flow_veh_per_hr", "avg_vehicles_in_zone", "peak_vehicles_in_zone",
                    "density_veh_per_km", "density_level"])

    def flush(end_s):
        nonlocal interval_start, interval_frames_seen
        dur = max(end_s - interval_start, 1e-6)
        for i, lane in enumerate(lanes):
            c = interval_counts[i]
            total = sum(c.values())
            avg = occ_sum[i] / max(interval_frames_seen, 1)
            length = lane.get("length_m")
            density = round(avg / (length / 1000.0), 1) if length else ""
            level = "Low" if avg < LOW_MAX else ("High" if avg >= HIGH_MIN else "Medium")
            csv_w.writerow([round(interval_start, 1), round(end_s, 1), lane["name"],
                            *[c[k] for k in CLASS_NAMES], total, round(total * 3600 / dur),
                            round(avg, 2), occ_peak[i], density, level])
            c.clear()
            occ_sum[i] = 0
            occ_peak[i] = 0
        csv_file.flush()
        interval_start = end_s
        interval_frames_seen = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t_s = frame_idx / fps

        # detect + track on the clean frame; draw only on the copy
        res = model(frame, classes=list(VEHICLES), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append(([x1, y1, x2 - x1, y2 - y1], float(b.conf[0]), int(b.cls[0])))
        tracks = tracker.update_tracks(dets, frame=frame)

        display = frame.copy()
        overlay = frame.copy()
        for i, poly in enumerate(polys):
            cv2.fillPoly(overlay, [poly], COLORS[i % len(COLORS)])
        display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)
        for i, poly in enumerate(polys):
            cv2.polylines(display, [poly], True, COLORS[i % len(COLORS)], 2)

        in_lane = [set() for _ in range(n)]
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            tid = t.track_id
            cls_id = t.get_det_class()
            if cls_id is not None:
                class_votes[tid][VEHICLES.get(int(cls_id), "car")] += 1
            label_cls = class_votes[tid].most_common(1)[0][0] if class_votes[tid] else "car"

            l, top, r, bot = [int(v) for v in t.to_ltrb()]
            foot = ((l + r) // 2, bot)   # bottom-centre = where the vehicle touches the road
            lane_i = point_lane(polys, foot)
            color = (200, 200, 200)

            if lane_i is not None:
                color = COLORS[lane_i % len(COLORS)]
                in_lane[lane_i].add(tid)
                frames_in_lane[(tid, lane_i)] += 1
                if tid not in counted and frames_in_lane[(tid, lane_i)] >= MIN_FRAMES_IN_LANE:
                    counted[tid] = lane_i
                    totals[lane_i][label_cls] += 1
                    interval_counts[lane_i][label_cls] += 1

            cv2.rectangle(display, (l, top), (r, bot), color, 2)
            cv2.putText(display, f"{label_cls} #{tid}", (l, max(top - 6, 12)), FONT, 0.5, color, 2)

        for i in range(n):
            occ_sum[i] += len(in_lane[i])
            occ_peak[i] = max(occ_peak[i], len(in_lane[i]))
        interval_frames_seen += 1
        if frame_idx % interval_frames == 0:
            flush(t_s)

        # HUD (dark panel, high contrast)
        panel_h = 28 + 24 * n
        cv2.rectangle(display, (0, 0), (430, panel_h), (0, 0, 0), -1)
        cv2.putText(display, f"t={t_s:6.1f}s", (8, 20), FONT, 0.6, (255, 255, 255), 2)
        for i, lane in enumerate(lanes):
            txt = f"{lane['name']}: counted {sum(totals[i].values())} | in zone {len(in_lane[i])}"
            cv2.putText(display, txt, (8, 44 + 24 * i), FONT, 0.6, COLORS[i % len(COLORS)], 2)

        if args.save:
            if vid_writer is None:
                h, w = display.shape[:2]
                vid_writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            vid_writer.write(display)
        if not args.no_show:
            show_scaled("Lane counter (q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if interval_frames_seen > 0:
        flush(frame_idx / fps)
    cap.release()
    if vid_writer:
        vid_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print("\n=== SURVEY SUMMARY ===")
    for i, lane in enumerate(lanes):
        parts = ", ".join(f"{k}: {totals[i][k]}" for k in CLASS_NAMES)
        print(f"{lane['name']}: total {sum(totals[i].values())}  ({parts})")
    print(f"Per-interval results saved to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lane-based vehicle counting and density")
    ap.add_argument("--source", default="sample.mp4", help="video file path or webcam index (0)")
    ap.add_argument("--setup", action="store_true", help="draw lanes on the first frame and save lanes.json")
    ap.add_argument("--lanes", default="lanes.json")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--out", default="counts.csv")
    ap.add_argument("--interval", type=float, default=10.0, help="seconds per reporting interval (900 = 15 min)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default="", help="optional path to save the annotated video, e.g. out.mp4")
    ap.add_argument("--no-show", action="store_true", help="run without a display window")
    args = ap.parse_args()

    if args.setup:
        setup_lanes(args.source, args.lanes)
    else:
        run(args)