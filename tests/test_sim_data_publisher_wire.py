"""Wire-level test: real roslibpy against a stub rosbridge websocket server.

The unit tests mock roslibpy entirely; this one exercises the real client and
asserts the rosbridge protocol frames that actually leave the process, so a
change in roslibpy's API or our message layout is caught without needing ROS.

Skipped unless both `roslibpy` and `websockets` are installed.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("roslibpy")
websockets = pytest.importorskip("websockets")

import sim_data_publisher as sdp


class StubRosbridge:
    """Minimal websocket server that records rosbridge ops it receives."""

    def __init__(self):
        self.ops = []
        self.port = None
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._stop = None

    async def _handler(self, ws):
        try:
            async for raw in ws:
                self.ops.append(json.loads(raw))
        except Exception:  # noqa: BLE001 - client disconnects end the handler
            pass

    async def _serve(self):
        async with websockets.serve(self._handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._stop = asyncio.Event()
            self._ready.set()
            await self._stop.wait()

    def __enter__(self):
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=10), "stub rosbridge failed to start"
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
        return False

    def ops_of(self, op_name):
        return [o for o in self.ops if o.get("op") == op_name]

    def wait_for(self, op_name, count, timeout=10.0):
        """Block until at least `count` ops of `op_name` have arrived."""
        deadline = threading.Event()
        step = 0.02
        waited = 0.0
        while waited < timeout:
            if len(self.ops_of(op_name)) >= count:
                return True
            deadline.wait(step)
            waited += step
        return False


def test_real_roslibpy_emits_expected_rosbridge_frames():
    state = [
        # rod_01: pos, quat (identity), linvel +y, angvel +z
        1.0, 2.0, 3.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 2.0,
    ]

    with StubRosbridge() as server:
        pub = sdp.RodStatePublisher(
            url=f"ws://127.0.0.1:{server.port}",
            rod_names=["rod_01"],
            stamp_source="sim",
            twist_frame="world",
            position_scale=1.0,
        )
        with pub:
            pub.publish_state(1.5, state)
            assert server.wait_for("publish", 1), (
                f"no publish frame arrived; got ops: {server.ops}"
            )

    advertises = server.ops_of("advertise")
    assert len(advertises) == 1
    assert advertises[0]["topic"] == "/tensegrity/rod_01/odom"
    assert advertises[0]["type"] == "nav_msgs/Odometry"

    publishes = server.ops_of("publish")
    assert len(publishes) == 1
    assert publishes[0]["topic"] == "/tensegrity/rod_01/odom"

    msg = publishes[0]["msg"]
    assert msg["header"]["frame_id"] == "world"
    assert msg["header"]["stamp"] == {"secs": 1, "nsecs": 500000000}
    assert msg["child_frame_id"] == "rod_01"
    assert msg["pose"]["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert msg["pose"]["pose"]["orientation"] == {
        "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
    }
    assert msg["twist"]["twist"]["linear"] == {"x": 0.0, "y": 1.0, "z": 0.0}
    assert msg["twist"]["twist"]["angular"] == {"x": 0.0, "y": 0.0, "z": 2.0}
    assert msg["pose"]["covariance"] == [0.0] * 36

    # The JSON must round-trip exactly as rosbridge would parse it.
    assert json.loads(json.dumps(msg)) == msg
