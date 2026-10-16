"""Phase 7.2: convert SACSoN (HuRoN) ROS1 bags into NoMaD's format with NoMaD's own, unchanged `train/process_bags.py`.

`process_bags.py` imports the ROS1 `rosbag` package, which needs ROS. This module installs a small stand-in under the
name `rosbag` (the four calls process_bags uses, read with the pure-Python `rosbags` package,
train/requirements-mapmad.txt) and then runs process_bags.py as its own `__main__`:

    cd /app/visualnav-transformer/train && python -m vint_train.mapmad.sacson_convert \
        -d sacson -i /data/sacson_raw -o /data/sacson -s 4.0

Arguments are process_bags.py's. Output per bag: <output>/<folder>_<bag>_<k>/{0.jpg, ..., traj_data.pkl} (full
160 x 120 fisheye frames at 4 Hz; training crops 4:3 and resizes later).
"""

import dataclasses
import runpy
import sys
import types
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

PROCESS_BAGS = Path(__file__).resolve().parents[2] / "process_bags.py"


class ROSBagException(Exception):
    """Same name as rosbag's exception, so process_bags' `except rosbag.ROSBagException` works."""


@dataclasses.dataclass(frozen=True)
class Time:
    """rospy.Time stand-in: nanoseconds with to_sec()."""

    ns: int

    def to_sec(self) -> float:
        return self.ns / 1e9


class Bag:
    """The part of rosbag.Bag that process_bags.py uses, read with rosbags (ROS1 Noetic message types)."""

    def __init__(self, path: Union[str, Path]) -> None:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore

        try:
            self._reader = Reader(Path(path))
            self._reader.open()
        except Exception as exc:  # unreadable / truncated bag
            raise ROSBagException(f"{path}: {exc}") from exc
        self._types = get_typestore(Stores.ROS1_NOETIC)

    def get_message_count(self, topic: str) -> int:
        return sum(c.msgcount for c in self._reader.connections if c.topic == topic)

    def get_start_time(self) -> float:
        return self._reader.start_time / 1e9

    def read_messages(self, topics: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, object, Time]]:
        """(topic, message, time) in time order; byte payloads (`data`) come as bytes, as rospy gives them."""
        cons = [c for c in self._reader.connections if topics is None or c.topic in topics]
        for con, t, raw in self._reader.messages(connections=cons):
            msg = self._types.deserialize_ros1(raw, con.msgtype)
            if isinstance(getattr(msg, "data", None), np.ndarray) and msg.data.dtype == np.uint8:
                msg = dataclasses.replace(msg, data=msg.data.tobytes())
            yield con.topic, msg, Time(t)

    def close(self) -> None:
        self._reader.close()


def install_shim() -> types.ModuleType:
    """Register this stand-in as the `rosbag` module."""
    mod = types.ModuleType("rosbag")
    mod.Bag, mod.ROSBagException = Bag, ROSBagException
    sys.modules["rosbag"] = mod
    return mod


def main(argv: Optional[List[str]] = None) -> None:
    install_shim()
    sys.argv = [str(PROCESS_BAGS)] + list(sys.argv[1:] if argv is None else argv)
    runpy.run_path(str(PROCESS_BAGS), run_name="__main__")


if __name__ == "__main__":
    main()
