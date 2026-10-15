import argparse
import math
import socket


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def wait(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description="Local trusted-model batching experiment; binds loopback only")
    parser.add_argument("model")
    parser.add_argument("--port", type=positive, default=8080)
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--wait-ms", type=wait, default=2.)
    parser.add_argument("--capacity", type=positive, default=128)
    parser.add_argument("--audit-limit", type=positive, default=4096)
    parser.add_argument("--socket-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capacity < args.batch_size or args.audit_limit < args.capacity or args.port > 65535:
        parser.error("Require batch <= capacity <= audit-limit and port <= 65535")
    from aiohttp import web
    from .server import create_app
    address = ({"sock": socket.socket(fileno=args.socket_fd)} if args.socket_fd is not None
               else {"host": "127.0.0.1", "port": args.port})
    web.run_app(create_app(args.model, batch_size=args.batch_size, wait_ms=args.wait_ms,
                           capacity=args.capacity, audit_limit=args.audit_limit), **address,
                access_log=None, print=None, handler_cancellation=True)
