"""
=============================================================
  AI-Powered Robotic Arm — Raspberry Pi Controller
  Flask Web Dashboard  +  YOLOv8 Vision  +  Serial to Arduino
  Author : Robotics & AI Engineer
=============================================================
  Install deps:
    pip install flask ultralytics opencv-python pyserial

  Run:
    python3 arm_controller.py
=============================================================
"""

import cv2
import serial
import threading
import time
import json
import os
import logging
from pathlib import Path
from collections import deque, defaultdict

from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arm")

# ─────────────────────────────────────────────────────────────
#  Config — edit these to match your setup
# ─────────────────────────────────────────────────────────────
SERIAL_PORT   = "COM8"           # Aapke screenshot ke mutabiq COM8
BAUD_RATE     = 115200
CAMERA_INDEX  = 0                # External USB camera ke liye 1 (ya 2)
MODEL_PATH    = r"C:\Users\muavi\Desktop\FYP_Project\best.pt" 
COORDS_FILE   = "arm_coords.json"
CONF_THRESH   = 0.55

# PASS criteria labels
REQUIRED_PASS_LABELS = {"Full", "Cap", "Label"}
REJECT_LABELS        = {"Underfilled", "Overfilled", "Deformed"}

# ─────────────────────────────────────────────────────────────
#  Shared State  (protected by locks where noted)
# ─────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # Servo angles (live / slider values)  [s0..s5]
        self.current_angles: list[int] = [90, 90, 90, 90, 90, 5]

        # Named coordinate presets [s0, s1, s2, s3, s4, s5]
        self.coords: dict[str, list[int]] = {
            "home":       [90, 90, 90, 90, 90, 5], # <--- 5 kar dein
            "pick":       [90, 90, 90, 90, 90, 90],
            "reject_box": [90, 90, 90, 90, 90, 90],
        }

        # Auto mode
        self.auto_mode: bool    = False
        self.arm_busy:  bool    = False

        # Vision results
        self.latest_frame        = None      # JPEG bytes
        self.latest_detections   = []        # list of {label, conf, box}
        self.latest_verdict      = "WAITING" # PASS / REJECT / WAITING
        self.detection_counts    = defaultdict(int)
        self.pass_count          = 0
        self.reject_count        = 0
        self.fps                 = 0.0

        # Serial
        self.serial_connected: bool = False
        self.serial_log: deque      = deque(maxlen=50)

state = SharedState()

# ─────────────────────────────────────────────────────────────
#  Coordinate persistence
# ─────────────────────────────────────────────────────────────
def load_coords():
    if Path(COORDS_FILE).exists():
        try:
            with open(COORDS_FILE) as f:
                data = json.load(f)
            with state.lock:
                state.coords.update(data)
            log.info("Loaded coordinates from %s", COORDS_FILE)
        except Exception as e:
            log.warning("Could not load coords: %s", e)

def save_coords():
    with state.lock:
        data = dict(state.coords)
    with open(COORDS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Coordinates saved to %s", COORDS_FILE)

# ─────────────────────────────────────────────────────────────
#  Serial Manager Thread
# ─────────────────────────────────────────────────────────────
class SerialManager(threading.Thread):
    """Owns the serial port; exposes send_angles() and a response queue."""

    def __init__(self):
        super().__init__(daemon=True, name="SerialMgr")
        self._ser: serial.Serial | None = None
        self._cmd_queue: deque[str]     = deque()
        self._response_event            = threading.Event()
        self._last_response             = ""
        self._lock                      = threading.Lock()

    # ── Public API ─────────────────────────────────────────────
    def send_angles(self, angles: list[int], wait_done=True, timeout=15):
        """Send angle string; optionally block until Arduino replies DONE."""
        cmd = ",".join(str(a) for a in angles) + "\n"
        with self._lock:
            self._cmd_queue.append(cmd)
        if wait_done:
            self._response_event.clear()
            return self._response_event.wait(timeout=timeout)
        return True

    def send_raw(self, cmd: str):
        with self._lock:
            self._cmd_queue.append(cmd + "\n")

    # ── Thread body ────────────────────────────────────────────
    def run(self):
        while True:
            try:
                self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(2)   # wait for Arduino reset
                self._ser.write(b"PING\n")
                log.info("Serial connected on %s", SERIAL_PORT)
                with state.lock:
                    state.serial_connected = True
                self._io_loop()
            except serial.SerialException as e:
                log.warning("Serial error: %s — retrying in 5 s", e)
                with state.lock:
                    state.serial_connected = False
            finally:
                if self._ser and self._ser.is_open:
                    self._ser.close()
            time.sleep(5)

    def _io_loop(self):
        while True:
            # Send pending commands
            with self._lock:
                cmd = self._cmd_queue.popleft() if self._cmd_queue else None
            if cmd:
                try:
                    self._ser.write(cmd.encode())
                    state.serial_log.append(f"TX → {cmd.strip()}")
                except serial.SerialException:
                    break

            # Read responses
            try:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="replace").strip()
                    if line:
                        state.serial_log.append(f"RX ← {line}")
                        self._last_response = line
                        if line in ("DONE", "PONG", "READY"):
                            self._response_event.set()
            except serial.SerialException:
                break

            time.sleep(0.005)

serial_mgr = SerialManager()

def move_sequentially(target_angles, delay=0.5):
    """
    Ek saath move karne ki bajaye bari bari joints ko move karta hai
    taake power supply par load na paray.
    """
    with state.lock:
        current = list(state.current_angles)
    
    for i in range(len(target_angles)):
        if current[i] != target_angles[i]:
            current[i] = target_angles[i]
            # Naya angle bhejte hain
            serial_mgr.send_angles(current, wait_done=True)
            # State update karte hain taake next iteration ko pata ho
            with state.lock:
                state.current_angles = list(current)
            # Chota sa gap taake power supply stable ho jaye
            time.sleep(delay)

# ─────────────────────────────────────────────────────────────
#  Arm Sequencer
# ─────────────────────────────────────────────────────────────
def arm_go_home():
    with state.lock:
        angles = list(state.coords["home"])
    serial_mgr.send_angles(angles)

def arm_reject_sequence():
    """Pick → Close Gripper → Reject Box → Open Gripper → Home."""
    with state.lock:
        pick       = list(state.coords["pick"])
        reject_box = list(state.coords["reject_box"])
        home       = list(state.coords["home"])
        state.arm_busy = True

    log.info("ARM: moving to PICK")
    serial_mgr.send_angles(pick)
    time.sleep(2) # Robot ko Pick position tak pohonchne dein

    # Close gripper — 5° (Safe Closed)
    close_gripper = list(pick)
    close_gripper[5] = 5 
    log.info("ARM: closing gripper")
    serial_mgr.send_angles(close_gripper)
    time.sleep(1.5) # Gripper ko botal pakadne ka waqt dein

    log.info("ARM: moving to REJECT BOX")
    serial_mgr.send_angles(reject_box)
    time.sleep(2) # Safely box tak pohonchne dein

    # Open gripper — 90° (Full Open)
    open_gripper = list(reject_box)
    open_gripper[5] = 90
    log.info("ARM: opening gripper")
    serial_mgr.send_angles(open_gripper)
    time.sleep(1) # Botal girne ka intezar karein

    log.info("ARM: returning HOME")
    serial_mgr.send_angles(home)
    time.sleep(2)

    with state.lock:
        state.arm_busy = False

def trigger_reject():
    t = threading.Thread(target=arm_reject_sequence, daemon=True, name="ArmSeq")
    t.start()

# ─────────────────────────────────────────────────────────────
#  Inspection Logic
# ─────────────────────────────────────────────────────────────
def evaluate_detections(detections: list[dict]) -> str:
    labels = {d["label"] for d in detections}
    if REQUIRED_PASS_LABELS.issubset(labels) and not labels.intersection(REJECT_LABELS):
        return "PASS"
    return "REJECT"

# ─────────────────────────────────────────────────────────────
#  YOLOv8 + Camera Thread
# ─────────────────────────────────────────────────────────────
class VisionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="Vision")

    def run(self):

        log.info("Loading YOLO model: %s", MODEL_PATH)
        model = YOLO(MODEL_PATH)
        log.info("YOLO loaded. Opening camera index %d", CAMERA_INDEX)
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_counter = 0
        fps_timer   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — retrying…")
                time.sleep(0.5)
                continue

            # ── Run inference ──────────────────────────────────
            results = model(frame, conf=CONF_THRESH, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "label": label, "conf": round(conf, 2),
                        "box": [x1, y1, x2, y2],
                    })
                    # Annotate frame
                    color = (0, 255, 0) if label in REQUIRED_PASS_LABELS else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2)

            verdict = evaluate_detections(detections)

            # ── Overlay verdict ────────────────────────────────
            v_color = (0, 220, 60) if verdict == "PASS" else \
                      (0, 0, 240) if verdict == "REJECT" else (180, 180, 180)
            cv2.putText(frame, verdict, (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, v_color, 3)

            # ── FPS counter ────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                with state.lock:
                    state.fps = fps_counter
                fps_counter = 0
                fps_timer   = time.time()

            cv2.putText(frame, f"FPS:{state.fps:.0f}", (530, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # ── Encode frame ────────────────────────────────────
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # ── Update shared state ─────────────────────────────
            with state.lock:
                state.latest_frame      = jpeg.tobytes()
                state.latest_detections = detections
                state.latest_verdict    = verdict
                for d in detections:
                    state.detection_counts[d["label"]] += 1

            # ── Auto-mode action ─────────────────────────────────
            with state.lock:
                auto   = state.auto_mode
                busy   = state.arm_busy
                do_act = auto and not busy and detections  # only act if something detected

            if do_act:
                with state.lock:
                    if verdict == "REJECT":
                        state.reject_count += 1
                    else:
                        state.pass_count += 1

                if verdict == "REJECT":
                    log.info("REJECT detected — triggering arm sequence")
                    trigger_reject()
                    time.sleep(3)   # debounce: skip frames while arm moves

        cap.release()

# ─────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Video stream generator ─────────────────────────────────────
def gen_frames():
    while True:
        with state.lock:
            frame = state.latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)   # ~25 fps cap

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Dashboard ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Status API (polled by JS) ──────────────────────────────────
@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify({
            "angles":          state.current_angles,
            "verdict":         state.latest_verdict,
            "detections":      state.latest_detections,
            "auto_mode":       state.auto_mode,
            "arm_busy":        state.arm_busy,
            "serial_connected":state.serial_connected,
            "pass_count":      state.pass_count,
            "reject_count":    state.reject_count,
            "detection_counts":dict(state.detection_counts),
            "fps":             state.fps,
            "serial_log":      list(state.serial_log)[-10:],
            "coords":          state.coords,
        })

# ── Manual servo control ───────────────────────────────────────
@app.route("/api/set_angles", methods=["POST"])
def api_set_angles():
    data   = request.json or {}
    angles = data.get("angles", [])
    if len(angles) != 6:
        return jsonify({"ok": False, "error": "Need exactly 6 angles"}), 400
    angles = [int(max(0, min(180, a))) for a in angles]
    if len(angles) > 5:
        if angles[5] > 90: angles[5] = 90
    with state.lock:
        state.current_angles = angles
    serial_mgr.send_angles(angles, wait_done=False)
    return jsonify({"ok": True, "angles": angles})

# ── Record current slider angles to a named pose ──────────────
@app.route("/api/record_pose", methods=["POST"])
def api_record_pose():
    data = request.json or {}
    pose = data.get("pose", "").lower()
    if pose not in state.coords:
        return jsonify({"ok": False, "error": f"Unknown pose '{pose}'"}), 400
    with state.lock:
        state.coords[pose] = list(state.current_angles)
    return jsonify({"ok": True, "pose": pose, "angles": state.coords[pose]})

# ── Persist coordinates to disk ────────────────────────────────
@app.route("/api/save_coords", methods=["POST"])
def api_save_coords():
    save_coords()
    return jsonify({"ok": True})

# ── Load coordinates from disk ─────────────────────────────────
@app.route("/api/load_coords", methods=["POST"])
def api_load_coords():
    load_coords()
    with state.lock:
        return jsonify({"ok": True, "coords": state.coords})

# ── Move arm to named pose ─────────────────────────────────────
@app.route("/api/go_pose", methods=["POST"])
def api_go_pose():
    data = request.json or {}
    pose = data.get("pose", "").lower()
    
    with state.lock:
        if pose not in state.coords:
            return jsonify({"ok": False, "error": "Unknown pose"}), 400
        target_angles = list(state.coords[pose])
        # Busy flag set karein taake koi aur command interfere na kare
        state.arm_busy = True

    # Purani logic (ek saath move) ko hata kar naye function ko call karein
    # Is se robot bari bari joints move karega
    threading.Thread(target=move_sequentially, args=(target_angles, 0.6), daemon=True).start()

    with state.lock:
        state.arm_busy = False
        
    return jsonify({"ok": True, "pose": pose, "angles": target_angles})

# ── Auto mode toggle ───────────────────────────────────────────
@app.route("/api/auto_mode", methods=["POST"])
def api_auto_mode():
    data = request.json or {}
    enable = bool(data.get("enable", False))
    with state.lock:
        state.auto_mode = enable
    if enable:
        arm_go_home()
    log.info("Auto mode: %s", "ENABLED" if enable else "DISABLED")
    return jsonify({"ok": True, "auto_mode": enable})

# ── Reset counters ─────────────────────────────────────────────
@app.route("/api/reset_stats", methods=["POST"])
def api_reset_stats():
    with state.lock:
        state.pass_count        = 0
        state.reject_count      = 0
        state.detection_counts  = defaultdict(int)
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    load_coords()

    # Start background threads
    serial_mgr.start()
    VisionThread().start()

    log.info("Flask dashboard starting on http://0.0.0.0:5000")
    # use_reloader=False is REQUIRED — avoids spawning duplicate threads
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)