"""Main entry point for Janggi AI server."""

import argparse
import os
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janggi AI Server")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to NNUE model file (e.g., models/nnue_smart_model.json)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Set model path as environment variable if provided
    if args.model:
        os.environ["NNUE_MODEL_PATH"] = args.model
        print(f"Using model: {args.model}")

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)

