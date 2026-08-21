#!/usr/bin/env python3
"""pybridge.py — the Python half of RunMat's ``py.*`` interop.

RunMat has no built-in Python interop (see RUNMAT.md), so this repo ships its
own. The MATLAB half lives in ``+py/`` and drives this script once per interop
call::

    python3 pybridge.py <request.json> <response.json>

Every call is a fresh interpreter: there is no daemon to supervise and no
handshake to get wrong. Objects that cannot be converted to MATLAB values
(estimators, fitted models, ...) are pickled into a session directory and
handed back as opaque handle ids, so they can be passed to a later call.

Wire format
-----------
Both directions encode every value as a JSON object tagged with ``pytag``, so
the MATLAB side never has to guess what ``jsondecode`` gave it:

    {"pytag": "none"}
    {"pytag": "num",    "v": 1.5}                  # or "special": nan/inf/-inf
    {"pytag": "str",    "v": "iris"}
    {"pytag": "bool",   "v": true}
    {"pytag": "array",  "file": "o0.bin", "dtype": "f8", "shape": [1, 38]}
    {"pytag": "list",   "items": [...]}            # also "tuple"
    {"pytag": "dict",   "items": [{"k": ..., "v": ...}, ...]}
    {"pytag": "module", "name": "compare"}
    {"pytag": "handle", "id": "h4f3c...", "type": "RandomForestClassifier"}

Numeric arrays travel as raw little-endian float64 (or uint8 for logicals) in
a side file, written column-major so MATLAB's ``fread`` + ``reshape`` reads it
back for free. Requests look like::

    {"op": "call", "target": <value>, "attr": "get_split",
     "args": [<value>, ...], "kwargs": {"name": <value>, ...},
     "call_dir": "...", "object_dir": "...", "sys_path": ["..."]}

and responses are ``{"ok": true, "value": <value>}`` or
``{"ok": false, "error": {"type": ..., "message": ..., "traceback": ...}}``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import pickle
import sys
import traceback
import types
import uuid

try:
    import numpy as np
except ImportError:  # numpy is optional until an array actually crosses
    np = None

TAG = "pytag"


class BridgeError(Exception):
    """Something the bridge itself cannot represent, as opposed to user code."""


class PreEncoded:
    """An op result that is already in wire form and must not be re-encoded."""

    def __init__(self, value):
        self.value = value


class Context:
    """Per-call scratch space: blob files here, pickled handles over there."""

    def __init__(self, call_dir, object_dir):
        self.call_dir = call_dir
        self.object_dir = object_dir
        self.blobs = 0

    # -- arrays ---------------------------------------------------------
    def write_array(self, arr):
        arr = np.asarray(arr)
        if arr.dtype.kind == "c":
            raise BridgeError(
                "complex arrays cannot cross the bridge yet; split into "
                "real(x) and imag(x) on the Python side"
            )
        if arr.dtype.kind not in "fiub":
            raise BridgeError(
                f"array of dtype {arr.dtype} cannot cross the bridge "
                "(only float, integer, and boolean arrays are supported)"
            )
        if arr.ndim == 0:
            return encode(arr.item(), self)
        dtype = "u1" if arr.dtype.kind == "b" else "f8"
        # MATLAB has no 1-D: hand a vector back as a 1-by-N row.
        shape = [1, arr.shape[0]] if arr.ndim == 1 else list(arr.shape)
        name = f"o{self.blobs}.bin"
        self.blobs += 1
        with open(os.path.join(self.call_dir, name), "wb") as fh:
            fh.write(arr.astype(dtype, copy=False).tobytes(order="F"))
        return {TAG: "array", "file": name, "dtype": dtype, "shape": shape}

    def read_array(self, value):
        if np is None:
            raise BridgeError("numpy is required to receive arrays from RunMat")
        path = os.path.join(self.call_dir, value["file"])
        with open(path, "rb") as fh:
            raw = fh.read()
        arr = np.frombuffer(raw, dtype=value["dtype"]).reshape(
            value["shape"], order="F"
        )
        if value["dtype"] == "u1":
            arr = arr.astype(bool)
        else:
            arr = arr.copy()  # frombuffer is read-only; sklearn dislikes that
        # MATLAB models every vector as 2-D. Unless the caller asked to keep
        # the shape (py.matrix), collapse row/column vectors to 1-D, which is
        # what numpy and sklearn expect.
        if value.get("squeeze", True) and arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        return arr

    # -- handles --------------------------------------------------------
    def store_handle(self, obj):
        hid = "h" + uuid.uuid4().hex[:12]
        path = os.path.join(self.object_dir, hid + ".pkl")
        try:
            with open(path, "wb") as fh:
                pickle.dump(obj, fh, protocol=4)
        except Exception as exc:  # unpicklable: say so instead of half-failing
            if os.path.exists(path):
                os.remove(path)
            raise BridgeError(
                f"cannot hand a {type(obj).__name__} back to RunMat: it is "
                f"neither convertible to a MATLAB value nor picklable ({exc}). "
                "Return something convertible, or keep it inside Python."
            )
        try:
            text = repr(obj)
        except Exception:
            text = f"<{type(obj).__name__}>"
        return {
            TAG: "handle",
            "id": hid,
            "type": type(obj).__name__,
            "repr": text[:200],
        }

    def load_handle(self, hid):
        path = os.path.join(self.object_dir, hid + ".pkl")
        if not os.path.exists(path):
            raise BridgeError(
                f"handle {hid} is gone — it was freed, or the interop session "
                "was reset since it was created"
            )
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------


def decode(value, ctx):
    """Wire value -> Python object."""
    if value is None:
        return None
    if not isinstance(value, dict) or TAG not in value:
        raise BridgeError(f"untagged value on the wire: {value!r}")
    kind = value[TAG]
    if kind == "none":
        return None
    if kind == "num":
        special = value.get("special")
        if special == "nan":
            return math.nan
        if special == "inf":
            return math.inf
        if special == "-inf":
            return -math.inf
        return value["v"]
    if kind == "str":
        return value["v"]
    if kind == "bool":
        return bool(value["v"])
    if kind == "array":
        return ctx.read_array(value)
    if kind in ("list", "tuple"):
        items = [decode(item, ctx) for item in value.get("items") or []]
        return items if kind == "list" else tuple(items)
    if kind == "dict":
        return {
            decode(pair["k"], ctx): decode(pair["v"], ctx)
            for pair in value.get("items") or []
        }
    if kind == "module":
        return importlib.import_module(value["name"])
    if kind == "handle":
        return ctx.load_handle(value["id"])
    if kind == "attr":
        return getattr(decode(value["of"], ctx), value["name"])
    raise BridgeError(f"unknown pytag {kind!r}")


def encode(obj, ctx):
    """Python object -> wire value."""
    if obj is None:
        return {TAG: "none"}
    if isinstance(obj, bool):
        return {TAG: "bool", "v": obj}
    if isinstance(obj, str):
        return {TAG: "str", "v": obj}
    if isinstance(obj, int):
        return {TAG: "num", "v": obj}
    if isinstance(obj, float):
        if math.isnan(obj):
            return {TAG: "num", "special": "nan"}
        if math.isinf(obj):
            return {TAG: "num", "special": "inf" if obj > 0 else "-inf"}
        return {TAG: "num", "v": obj}
    if isinstance(obj, types.ModuleType):
        return {TAG: "module", "name": obj.__name__}
    if np is not None:
        if isinstance(obj, np.generic):
            return encode(obj.item(), ctx)
        if isinstance(obj, np.ndarray):
            return ctx.write_array(obj)
    if isinstance(obj, (list, tuple)):
        return {
            TAG: "list" if isinstance(obj, list) else "tuple",
            "items": [encode(item, ctx) for item in obj],
        }
    if isinstance(obj, dict):
        return {
            TAG: "dict",
            "items": [
                {"k": encode(k, ctx), "v": encode(v, ctx)} for k, v in obj.items()
            ],
        }
    if isinstance(obj, (bytes, bytearray)):
        return {TAG: "str", "v": obj.decode("utf-8", "replace")}
    return ctx.store_handle(obj)


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------


def op_ping(request, ctx):
    info = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "prefix": sys.prefix,
        "numpy": getattr(np, "__version__", None),
    }
    for name in ("sklearn", "matplotlib", "skore"):
        try:
            info[name] = importlib.import_module(name).__version__
        except Exception:
            info[name] = None
    return info


def op_import(request, ctx):
    module = importlib.import_module(request["name"])
    return module


def op_reload(request, ctx):
    module = decode(request["target"], ctx)
    if not isinstance(module, types.ModuleType):
        raise BridgeError("py.importlib.reload expects a module")
    return importlib.reload(module)


def op_call(request, ctx):
    target = decode(request["target"], ctx)
    attr = request.get("attr") or ""
    callee = getattr(target, attr) if attr else target
    if not callable(callee):
        where = f"{attr!r} of {type(target).__name__}" if attr else repr(target)
        raise BridgeError(f"{where} is not callable")
    args = [decode(a, ctx) for a in request.get("args") or []]
    kwargs = {k: decode(v, ctx) for k, v in (request.get("kwargs") or {}).items()}
    return callee(*args, **kwargs)


def op_getattr(request, ctx):
    target = decode(request["target"], ctx)
    value = getattr(target, request["attr"])
    if callable(value):
        # Functions and bound methods are rarely picklable and never useful as
        # MATLAB data. Tell the MATLAB side to wrap it as a bound PyObject it
        # can call later, which costs no pickling at all.
        return PreEncoded({TAG: "callable", "name": request["attr"]})
    return value


def op_eval(request, ctx):
    names = {k: decode(v, ctx) for k, v in (request.get("names") or {}).items()}
    scope = {"__builtins__": __builtins__}
    scope.update(names)
    code = request["code"]
    if request.get("statements"):
        exec(code, scope)  # noqa: S102 — the user asked for exec
        out = request.get("output")
        return scope.get(out) if out else None
    return eval(code, scope)  # noqa: S307 — ditto


def op_free(request, ctx):
    freed = 0
    for hid in request.get("ids") or []:
        path = os.path.join(ctx.object_dir, hid + ".pkl")
        if os.path.exists(path):
            os.remove(path)
            freed += 1
    return freed


OPS = {
    "ping": op_ping,
    "import": op_import,
    "reload": op_reload,
    "call": op_call,
    "getattr": op_getattr,
    "eval": op_eval,
    "free": op_free,
}


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: pybridge.py <request.json> <response.json>\n")
        return 2
    request_path, response_path = argv[1], argv[2]
    with open(request_path, "r", encoding="utf-8") as fh:
        request = json.load(fh)

    for entry in reversed(request.get("sys_path") or []):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    ctx = Context(request["call_dir"], request["object_dir"])
    try:
        op = request.get("op", "call")
        if op not in OPS:
            raise BridgeError(f"unknown op {op!r}")
        result = OPS[op](request, ctx)
        if isinstance(result, PreEncoded):
            response = {"ok": True, "value": result.value}
        else:
            response = {"ok": True, "value": encode(result, ctx)}
    except BaseException as exc:  # user code failing is a normal outcome here
        response = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

    sys.stdout.flush()
    with open(response_path, "w", encoding="utf-8") as fh:
        json.dump(response, fh, allow_nan=False)
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
