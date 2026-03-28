"""
Camera-Based Pedestrian Location Detection and LED Warning System
Raspberry Pi 4 + Camera Module + 3x LEDs on GPIO 17, 27, 22

Wiring:
  GPIO 17 --> 220Ω resistor --> LED1 (Zone 1: curb/sidewalk) --> GND
  GPIO 27 --> 220Ω resistor --> LED2 (Zone 2: middle of street) --> GND
  GPIO 22 --> 220Ω resistor --> LED3 (Zone 3: opposite side) --> GND

How it works:
  Each video frame is downscaled to 320x240 before running the HOG pedestrian
  detector. The largest detected bounding box determines which horizontal third
  of the frame the pedestrian occupies (zone 1/2/3). A hysteresis counter
  (CONFIRM_FRAMES) prevents flickering: the LED only switches after the same
  zone is detected in N consecutive frames.
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

# BCM pin numbers for each zone LED
GPIO_PINS = {
    1: 17,   # Zone 1 — sidewalk / entry
    2: 27,   # Zone 2 — middle of street
    3: 22,   # Zone 3 — opposite side
}

# BGR colors used to draw bounding boxes and status bars per zone
ZONE_COLORS = {
    1: (0, 200, 80),    # green
    2: (0, 165, 255),   # orange
    3: (0, 60, 220),    # red
}

# Human-readable zone labels shown in the preview overlay
ZONE_NAMES = {
    1: "Zone 1: Sidewalk",
    2: "Zone 2: Mid-street",
    3: "Zone 3: Far side",
}

# HOG detector tuning — adjust these to trade speed vs. sensitivity
HOG_WIN_STRIDE   = (8, 8)   # pixels between detector windows; smaller = slower but more thorough
HOG_PADDING      = (4, 4)   # extra pixels around each window before classification
HOG_SCALE        = 1.05     # image pyramid scale factor; closer to 1.0 = more scales = slower
HOG_HIT_THRESH   = 0.0      # SVM decision threshold; raise to reduce false positives

# Number of consecutive frames a zone must be detected before the LED switches.
# Higher = less flickering, but slower response to pedestrian movement.
CONFIRM_FRAMES   = 3

# Camera settings
CAMERA_INDEX     = 0        # /dev/video0 on Linux; change for USB cameras
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
    """Configure all LED pins as outputs, initially LOW (off)."""
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
    """Turn off all LEDs and release GPIO resources before exit."""
    if not GPIO_AVAILABLE:
        return
    set_led(None)
    GPIO.cleanup()
    log.info("GPIO cleaned up")


# ---------------------------------------------------------------------------
# Zone detection
# ---------------------------------------------------------------------------

def get_zone(cx: int, frame_width: int) -> int:
    """Map a bounding-box centroid x-coordinate to a zone (1, 2, or 3).

    The frame is divided into three equal vertical strips:
      Zone 1 — left third   (pedestrian near the curb / entry side)
      Zone 2 — middle third (pedestrian crossing mid-street)
      Zone 3 — right third  (pedestrian on the far side)
    """
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

    Largest area is used as a proxy for confidence: closer / more visible
    pedestrians produce bigger bounding boxes with the HOG detector.
    """
    if len(detections) == 0:
        return None, None

    # detections may come back as a tuple-of-arrays from detectMultiScale
    rects = detections[0] if isinstance(detections, tuple) else detections
    best  = max(rects, key=lambda r: r[2] * r[3])   # maximise w*h
    x, y, w, h = best
    cx = x + w // 2   # horizontal centre of the bounding box
    zone = get_zone(cx, frame_width)
    return zone, (x, y, w, h)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame, zone: int | None, bbox: tuple | None,
                 active_led: int | None, fps: float):
    """Annotate *frame* in-place with zone lines, bounding box, LED bar, and FPS."""
    h, w = frame.shape[:2]
    third = w // 3

    # Vertical lines dividing the frame into the three zones
    for x in (third, third * 2):
        cv2.line(frame, (x, 0), (x, h), (200, 200, 200), 1, cv2.LINE_AA)

    # Zone name labels centred in each column at the top
    for z in (1, 2, 3):
        label_x = (z - 1) * third + third // 2
        cv2.putText(frame, ZONE_NAMES[z], (label_x - 60, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    # Bounding box and zone label around the detected pedestrian
    if bbox is not None and zone is not None:
        x, y, bw, bh = bbox
        color = ZONE_COLORS[zone]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = ZONE_NAMES[zone]
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Filled background rectangle so the label is readable over any background
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # LED status bar across the bottom — one coloured block per zone
    for z in (1, 2, 3):
        bar_x = (z - 1) * third
        is_on  = (z == active_led)
        color  = ZONE_COLORS[z] if is_on else (60, 60, 60)   # dim grey when off
        cv2.rectangle(frame, (bar_x, h - 28), (bar_x + third, h), color, -1)
        tag = f"LED {z}  ON" if is_on else f"LED {z}  off"
        cv2.putText(frame, tag, (bar_x + 8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255) if is_on else (140, 140, 140),
                    1, cv2.LINE_AA)

    # Exponential-moving-average FPS in the top-right corner
    cv2.putText(frame, f"{fps:.1f} fps", (w - 72, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(show_preview: bool = True, camera_index: int = CAMERA_INDEX):
    """
    Main detection loop.

    1. Grab a frame from the camera.
    2. Downscale it to 320x240 and run the HOG pedestrian detector.
    3. Scale the detected bounding boxes back to full resolution.
    4. Determine which zone the largest detection falls in.
    5. Apply hysteresis: only update the active LED after CONFIRM_FRAMES
       consecutive frames agree on the same zone.
    6. Optionally display an annotated preview window.
    """
    # Load OpenCV's built-in HOG + Linear SVM pedestrian detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    log.info("HOG detector ready")

    # Open the camera and request the configured resolution
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        log.error("Cannot open camera (index %d)", camera_index)
        sys.exit(1)
    log.info("Camera opened at %dx%d", FRAME_WIDTH, FRAME_HEIGHT)

    gpio_setup()

    # --- Hysteresis state ---
    pending_zone  = None   # zone detected in the most recent frame
    confirm_count = 0      # how many consecutive frames have reported pending_zone
    active_led    = None   # zone whose LED is currently ON (None = all off)

    # Exponential moving average for FPS display
    fps    = 0.0
    t_prev = time.perf_counter()

    log.info("Detection running — press Q in preview window (or Ctrl+C) to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame — retrying…")
                time.sleep(0.05)
                continue

            # Downscale before detection: HOG is O(n) in pixel count, so
            # halving resolution gives ~4x speed-up with acceptable accuracy.
            small = cv2.resize(frame, (320, 240))
            # Keep the x-scale factor so we can map boxes back to full resolution
            scale = frame.shape[1] / small.shape[1]

            # Run HOG detector on the small frame
            rects, _ = hog.detectMultiScale(
                small,
                winStride    = HOG_WIN_STRIDE,
                padding      = HOG_PADDING,
                scale        = HOG_SCALE,
                hitThreshold = HOG_HIT_THRESH,
            )

            # Map bounding boxes back to original frame coordinates
            if len(rects):
                rects = (rects * scale).astype(int)

            detected_zone, bbox = best_detection(rects, frame.shape[1])

            # Hysteresis: increment counter if the same zone persists,
            # otherwise reset and start counting from the new zone.
            if detected_zone == pending_zone and detected_zone is not None:
                confirm_count += 1
            else:
                pending_zone  = detected_zone
                confirm_count = 1

            # Switch the LED only after the zone has been stable for CONFIRM_FRAMES
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
                # Edge case: no zone detected but counter was already saturated
                active_led = None
                set_led(None)

            # Turn LEDs off once the pedestrian has been absent for CONFIRM_FRAMES
            if detected_zone is None:
                if confirm_count >= CONFIRM_FRAMES and active_led is not None:
                    active_led = None
                    set_led(None)
                    log.info("Pedestrian left frame — all LEDs OFF")

            # Update exponential moving average FPS (α = 0.1)
            t_now  = time.perf_counter()
            fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            # Draw and show the annotated preview frame
            if show_preview:
                annotated = draw_overlay(
                    frame.copy(), detected_zone, bbox, active_led, fps
                )
                cv2.imshow("Pedestrian Detector", annotated)
                # waitKey(1) pumps the GUI event loop; returns the pressed key
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
