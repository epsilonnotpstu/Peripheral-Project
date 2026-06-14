"""
Serial simulator for development without hardware.

Reads MIT-BIH test CSV and outputs lines to stdout in the ESP8266
serial protocol format. Pipe into the application or use as a
standalone test of the signal processing pipeline.

Usage:
    # Run the server in simulate mode (default in development):
    python run.py

    # Or manually test the protocol:
    python scripts/simulate_serial.py | head -50

    # Speed modes:
    python scripts/simulate_serial.py --speed 2.0   # 2x realtime
    python scripts/simulate_serial.py --no-sleep     # as fast as possible
    python scripts/simulate_serial.py --rows 1000    # limit rows
"""

import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def simulate(
    csv_path: str,
    fs: int = 125,
    speed: float = 1.0,
    no_sleep: bool = False,
    max_rows: int = None,
    add_motion: bool = True,
) -> None:
    """
    Read MIT-BIH test CSV and stream lines to stdout in ESP8266 format.

    Protocol: ECG,<ts_ms>,<adc_0-1023>[,<ax100>,<ay100>,<az100>]

    Args:
        csv_path: Path to mitbih_test.csv
        fs: Sampling rate in Hz (default 125)
        speed: Speed multiplier (default 1.0 = realtime)
        no_sleep: If True, output as fast as possible (for testing)
        max_rows: Stop after this many CSV rows (one row = 187 samples)
        add_motion: Simulate MPU6050 accelerometer data
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"# Simulating serial output from: {csv_path}", file=sys.stderr)
    print(f"# Protocol: ECG,<ts_ms>,<adc>[,<ax100>,<ay100>,<az100>]", file=sys.stderr)
    print(f"# Speed: {speed}x, motion: {add_motion}", file=sys.stderr)

    df = pd.read_csv(csv_path, header=None)
    if max_rows:
        df = df.head(max_rows)

    sample_interval = (1.0 / fs) / speed   # sleep time between samples
    start_time = int(time.time() * 1000)
    sample_idx = 0

    # Simulate mild motion for some beats (1 in 20 beats)
    motion_active = False
    beats_until_motion_change = 20

    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        samples = row.iloc[:187].values.astype(float)

        # Toggle simulated motion
        beats_until_motion_change -= 1
        if beats_until_motion_change <= 0:
            motion_active = not motion_active
            beats_until_motion_change = np.random.randint(10, 50)

        for val in samples:
            ts_ms = start_time + int(sample_idx * (1000.0 / fs))
            adc = int(np.clip(val, 0.0, 1.0) * 1023)

            if add_motion:
                # Simulate accelerometer: 1g static gravity + noise during motion
                noise = np.random.normal(0, 0.15 if motion_active else 0.01, 3)
                ax = int((0.0 + noise[0]) * 100)
                ay = int((0.0 + noise[1]) * 100)
                az = int((1.0 + noise[2]) * 100)
                line = f"ECG,{ts_ms},{adc},{ax},{ay},{az}"
            else:
                line = f"ECG,{ts_ms},{adc}"

            print(line, flush=True)
            sample_idx += 1

            if not no_sleep:
                time.sleep(sample_interval)

    print(f"# Simulation complete: {sample_idx} samples, {row_idx + 1} beats", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate ESP8266 serial output")
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "mitbih dataset", "mitbih_test.csv"),
        help="Path to test CSV file",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--no-sleep", action="store_true", help="Output as fast as possible")
    parser.add_argument("--rows", type=int, default=None, help="Max CSV rows to replay")
    parser.add_argument("--no-motion", action="store_true", help="Disable motion simulation")
    args = parser.parse_args()

    simulate(
        csv_path=args.csv,
        speed=args.speed,
        no_sleep=args.no_sleep,
        max_rows=args.rows,
        add_motion=not args.no_motion,
    )


# aldfasldfjsadl