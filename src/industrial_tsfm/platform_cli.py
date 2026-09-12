from __future__ import annotations

import argparse
import json
from pathlib import Path


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            'Platform API dependencies are missing. Install with: pip install -e ".[platform]"'
        ) from exc

    from .platform.opcua_api import create_live_app

    app = create_live_app(args.artifact_root, args.workspace_root)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def _capabilities(_: argparse.Namespace) -> None:
    try:
        from .platform.api import platform_capabilities
    except ImportError as exc:
        raise SystemExit(
            'Platform API dependencies are missing. Install with: pip install -e ".[platform]"'
        ) from exc
    payload = platform_capabilities()
    payload["live_extensions"] = {
        "opcua": {
            "status": "read_only_live_v1",
            "operations": ["browse", "snapshot_read", "subscribe", "runtime_status", "runtime_stop"],
            "optional_dependency": "industrial-tsfm[opcua]",
            "writes_enabled": False,
        },
        "mqtt": {"status": "contract_only"},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _list_applications(args: argparse.Namespace) -> None:
    try:
        from .platform.api import ApplicationArtifactStore
    except ImportError as exc:
        raise SystemExit(
            'Platform API dependencies are missing. Install with: pip install -e ".[platform]"'
        ) from exc
    store = ApplicationArtifactStore(Path(args.artifact_root))
    print(json.dumps(store.list_applications(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="IndusTSFM industrial platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="serve the product API with live extensions")
    serve.add_argument("--artifact-root", default="results")
    serve.add_argument("--workspace-root", default=".industsfm/workspace")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_serve)

    capabilities = subparsers.add_parser("capabilities", help="print product capabilities")
    capabilities.set_defaults(handler=_capabilities)

    listing = subparsers.add_parser("list", help="list materialized applications")
    listing.add_argument("--artifact-root", default="results")
    listing.set_defaults(handler=_list_applications)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
