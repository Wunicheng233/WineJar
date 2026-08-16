#!/usr/bin/env python3
"""Keep GroundingDINO and SAM2 loaded and serve local frame-inference requests."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
import importlib.util
from pathlib import Path
from typing import Any

import torch


OFFLINE_SCRIPT = Path(__file__).with_name("01_offline_grounded_sam2.py")
OFFLINE_SPEC = importlib.util.spec_from_file_location("offline_grounded_sam2_service", OFFLINE_SCRIPT)
if OFFLINE_SPEC is None or OFFLINE_SPEC.loader is None:
    raise ImportError(f"Cannot load vision pipeline from {OFFLINE_SCRIPT}")
OFFLINE = importlib.util.module_from_spec(OFFLINE_SPEC)
sys.modules[OFFLINE_SPEC.name] = OFFLINE
OFFLINE_SPEC.loader.exec_module(OFFLINE)

DEFAULT_SOCKET = Path("/tmp/winejar_grounded_sam2.sock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--model-id", default=OFFLINE.DEFAULT_MODEL_ID)
    parser.add_argument("--sam2-model-id", default=OFFLINE.DEFAULT_SAM2_MODEL_ID)
    parser.add_argument("--sam2-config", default=OFFLINE.DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=OFFLINE.DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def read_request(connection: socket.socket) -> dict[str, Any]:
    file = connection.makefile("r", encoding="utf-8")
    line = file.readline()
    if not line:
        raise RuntimeError("Client closed the connection without a request")
    request = json.loads(line)
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    return request


def write_response(connection: socket.socket, response: dict[str, Any]) -> None:
    payload = json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n"
    connection.sendall(payload.encode("utf-8"))


def main() -> int:
    args = parse_args()
    socket_path = args.socket.expanduser().resolve()
    if socket_path.exists():
        socket_path.unlink()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = args.sam2_checkpoint.expanduser().resolve() if args.sam2_checkpoint else None
    print(f"Loading GroundingDINO and SAM2 on {device}", flush=True)
    started_loading = time.monotonic()
    processor, detector = OFFLINE.load_grounding_dino(
        model_id=args.model_id,
        device=device,
        local_files_only=not args.allow_model_download,
    )
    sam2_predictor = OFFLINE.build_sam2_predictor(
        device=device,
        config_file=args.sam2_config,
        checkpoint=checkpoint,
        model_id=args.sam2_model_id,
    )
    loading_seconds = time.monotonic() - started_loading

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    should_stop = False
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(4)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "socket": str(socket_path),
                    "device": str(device),
                    "model_loading_seconds": loading_seconds,
                }
            ),
            flush=True,
        )

        try:
            while not should_stop:
                connection, _ = server.accept()
                with connection:
                    try:
                        request = read_request(connection)
                        command = str(request.get("command", "infer"))
                        request_id = request.get("request_id")
                        if command == "ping":
                            write_response(
                                connection,
                                {
                                    "ok": True,
                                    "request_id": request_id,
                                    "status": "ready",
                                    "device": str(device),
                                    "model_loading_seconds": loading_seconds,
                                },
                            )
                            continue
                        if command == "shutdown":
                            write_response(connection, {"ok": True, "request_id": request_id})
                            should_stop = True
                            continue
                        if command != "infer":
                            raise ValueError(f"Unsupported command: {command}")

                        frame_dir = Path(str(request["frame_dir"])).expanduser().resolve()
                        if not frame_dir.is_dir():
                            raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
                        result = OFFLINE.process_saved_frame(
                            frame_dir=frame_dir,
                            text_prompt=str(request["text_prompt"]),
                            processor=processor,
                            detector=detector,
                            sam2_predictor=sam2_predictor,
                            device=device,
                            model_id=args.model_id,
                            sam2_model_id=args.sam2_model_id,
                            sam2_config=args.sam2_config,
                            sam2_checkpoint=checkpoint,
                            box_threshold=float(request.get("box_threshold", 0.45)),
                            text_threshold=float(request.get("text_threshold", 0.15)),
                            output_min_score=float(request.get("output_min_score", 0.45)),
                        )
                        write_response(
                            connection,
                            {
                                "ok": True,
                                "request_id": request_id,
                                "frame_dir": str(frame_dir),
                                "detection_count": len(result.get("detections", [])),
                                "timing_seconds": result.get("timing_seconds", {}),
                            },
                        )
                    except Exception as exc:
                        traceback.print_exc(file=sys.stderr)
                        write_response(
                            connection,
                            {
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
        except KeyboardInterrupt:
            pass
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()
        print("Vision service stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
