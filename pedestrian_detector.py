"""
Camera-Based Pedestrian Location Detection and LED Warning System
Raspberry Pi 4 + Camera Module + 3x LEDs on GPIO 17, 27, 22

Wiring:
  GPIO 17 --> 220Ω resistor --> LED1 (Zone 1: curb/sidewalk) --> GND
  GPIO 27 --> 220Ω resistor --> LED2 (Zone 2: middle of street) --> GND
  GPIO 22 --> 220Ω resistor --> LED3 (Zone 3: opposite side) --> GND
"""

import cv2
import time
import sys
import logging
import argparse

# GPIO is only available on Raspberry Pi; fall back to a stub for development
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[INFO] RPi.GPIO not found — running in preview-only mode (no LED output)")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GPIO_PINS = {
    1: 17,   # Zone 1 — sidewalk / entry
    2: 27,   # Zone 2 — middle of street
    3: 22,   # Zone 3 — opposite side
}

ZONE_COLORS = {
    1: (0, 200, 80),    # green
    2: (0, 165, 255),   # orange
    3: (0, 60, 220),    # red
}

ZONE_NAMES = {
    1: "Zone 1: Sidewalk",
    2: "Zone 2: Mid-street",
    3: "Zone 3: Far side",
}

# Detection parameters
HOG_WIN_STRIDE   = (8, 8)
HOG_PADDING      = (4, 4)
HOG_SCALE        = 1.05
HOG_HIT_THRESH   = 0.0     # lower = more detections, higher = stricter

# Hysteresis: require this many consecutive frames before switching LED
CONFIRM_FRAMES   = 3

# Camera
CAMERA_INDEX     = 0
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pedestrian_detector")


# ---------------------------------------------------------------------------
# GPIO helpers
# ---------------------------------------------------------------------------

def gpio_setup():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in GPIO_PINS.values():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    log.info("GPIO initialised — pins %s", list(GPIO_PINS.values()))


def set_led(zone: int | None):
    """Turn on the LED for *zone* (1/2/3) and turn off the others.
    Pass None to turn off all LEDs."""
    if not GPIO_AVAILABLE:
        return
    for z, pin in GPIO_PINS.items():
        GPIO.output(pin, GPIO.HIGH if z == zone else GPIO.LOW)


def gpio_cleanup():
    if not GPIO_AVAILABLE:
        return
    set_led(None)
    GPIO.cleanup()
    log.info("GPIO cleaned up")


# ---------------------------------------------------------------------------
# Zone detection
# ---------------------------------------------------------------------------

def get_zone(cx: int, frame_width: int) -> int:
    """Map a bounding-box centroid x-coordinate to a zone (1, 2, or 3)."""
    third = frame_width / 3
    if cx < third:
        return 1
    elif cx < 2 * third:
        return 2
    else:
        return 3


def best_detection(detections, frame_width: int) -> tuple[int | None, tuple | None]:
    """
    From all detected bounding boxes pick the largest (most confident) one.
    Returns (zone, bbox) or (None, None) if no detections.
    """
    if len(detections) == 0:
        return None, None

    # Unpack (x, y, w, h) and pick box with largest area
    rects = detections[0] if isinstance(detections, tuple) else detections
    best  = max(rects, key=lambda r: r[2] * r[3])
    x, y, w, h = best
    cx = x + w // 2
    zone = get_zone(cx, frame_width)
    return zone, (x, y, w, h)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame, zone: int | None, bbox: tuple | None,
                 active_led: int | None, fps: float):
    h, w = frame.shape[:2]
    third = w // 3

    # Zone divider lines
    for x in (third, third * 2):
        cv2.line(frame, (x, 0), (x, h), (200, 200, 200), 1, cv2.LINE_AA)

    # Zone labels along the top
    for z in (1, 2, 3):
        label_x = (z - 1) * third + third // 2
        cv2.putText(frame, ZONE_NAMES[z], (label_x - 60, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    # Pedestrian bounding box
    if bbox is not None and zone is not None:
        x, y, bw, bh = bbox
        color = ZONE_COLORS[zone]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = ZONE_NAMES[zone]
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # LED status bar (bottom)
    for z in (1, 2, 3):
        bar_x = (z - 1) * third
        is_on  = (z == active_led)
        color  = ZONE_COLORS[z] if is_on else (60, 60, 60)
        cv2.rectangle(frame, (bar_x, h - 28), (bar_x + third, h), color, -1)
        tag = f"LED {z}  ON" if is_on else f"LED {z}  off"
        cv2.putText(frame, tag, (bar_x + 8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255) if is_on else (140, 140, 140),
                    1, cv2.LINE_AA)

    # FPS counter
    cv2.putText(frame, f"{fps:.1f} fps", (w - 72, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(show_preview: bool = True, camera_index: int = CAMERA_INDEX):
    # Initialise HOG pedestrian detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    log.info("HOG detector ready")

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        log.error("Cannot open camera (index %d)", camera_index)
        sys.exit(1)
    log.info("Camera opened at %dx%d", FRAME_WIDTH, FRAME_HEIGHT)

    gpio_setup()

    # Hysteresis state
    pending_zone  = None     # zone seen in recent frames
    confirm_count = 0        # consecutive frames with same zone
    active_led    = None     # zone whose LED is currently ON

    fps      = 0.0
    t_prev   = time.perf_counter()

    log.info("Detection running — press Q in preview window (or Ctrl+C) to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame — retrying…")
                time.sleep(0.05)
                continue

            # Resize for faster inference (detect on small frame, draw on original)
            small  = cv2.resize(frame, (320, 240))
            scale  = frame.shape[1] / small.shape[1]   # x-scale back to original

            # Run HOG detector
            rects, _ = hog.detectMultiScale(
                small,
                winStride   = HOG_WIN_STRIDE,
                padding     = HOG_PADDING,
                scale       = HOG_SCALE,
                hitThreshold= HOG_HIT_THRESH,
            )

            # Scale bounding boxes back to full resolution
            if len(rects):
                rects = (rects * scale).astype(int)

            detected_zone, bbox = best_detection(rects, frame.shape[1])

            # Hysteresis — confirm zone across CONFIRM_FRAMES frames
            if detected_zone == pending_zone and detected_zone is not None:
                confirm_count += 1
            else:
                pending_zone  = detected_zone
                confirm_count = 1

            if confirm_count >= CONFIRM_FRAMES:
                if active_led != detected_zone:
                    active_led = detected_zone
                    set_led(active_led)
                    if active_led:
                        log.info("Pedestrian confirmed in %s → LED %d ON",
                                 ZONE_NAMES[active_led], active_led)
                    else:
                        log.info("No pedestrian — all LEDs OFF")
            elif detected_zone is None and confirm_count >= CONFIRM_FRAMES:
                active_led = None
                set_led(None)

            # When no detection for CONFIRM_FRAMES frames, turn LEDs off
            if detected_zone is None:
                if confirm_count >= CONFIRM_FRAMES and active_led is not None:
                    active_led = None
                    set_led(None)
                    log.info("Pedestrian left frame — all LEDs OFF")

            # FPS
            t_now = time.perf_counter()
            fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # Preview window
            if show_preview:
                annotated = draw_overlay(
                    frame.copy(), detected_zone, bbox, active_led, fps
                )
                cv2.imshow("Pedestrian Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Q pressed — exiting")
                    break

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        gpio_cleanup()
        log.info("Done")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pedestrian zone detector with LED output"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable the OpenCV preview window (headless mode)"
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Camera device index (default: {CAMERA_INDEX})"
    )
    args = parser.parse_args()

    run(show_preview=not args.no_preview, camera_index=args.camera)
