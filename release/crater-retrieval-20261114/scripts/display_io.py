"""Small exclusive publication helpers retained from the original writers."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new_atomic(path, record):
    """Publish complete bytes exclusively; a pre-existing destination is untouched."""
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".display-", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write_completion(path, record):
    """The exclusive link commits completion; later cleanup cannot revoke it."""
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    name, committed = None, False
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
        committed = True
    finally:
        if name is not None:
            try:
                os.unlink(name)
            except Exception:
                # A residual temporary name does not invalidate a committed receipt.
                if not committed:
                    raise


def write_bytes_new(path, raw):
    """Binary counterpart of the existing exclusive JSON publication helper."""
    path = Path(path); staging = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".geography-", delete=False) as stream:
            staging = Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(staging, path)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
