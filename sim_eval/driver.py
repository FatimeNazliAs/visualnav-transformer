"""The four numbers that decide how the policy steers.

`navigate.py` takes them as command-line flags; this harness takes them as a
config block, for a reason that is about evidence rather than convenience.

Until P4's architecture pass they were module constants, reachable only by
editing the source. That made them the only dials in the package with neither a
config section nor a column in the results table — every other one
(`success_radius_m`, `timeout_slack`, the trail spacing) has both. A table
written under a changed value was indistinguishable from one written under the
default, which is exactly the ambiguity the next two phases cannot afford:

  * **P5** is allowed to "sanity-tune only what fidelity allows" (plan §9), so
    it will move these numbers, repeatedly.
  * **P6** compares two checkpoints under the fairness protocol (plan §7),
    which requires every arm to face identical settings.

The defaults below are unchanged and are meant to stay that way — plan decision
E is to mirror the real LoCoBot and tune nothing. `label()` is what lets a row
*prove* that instead of asserting it.

This lives apart from `nomad_policy` so that reading or validating the
configuration costs nothing: that module imports torch, diffusers and the whole
NoMaD stack, and a config object has no business being behind them. Every other
config class in this package is importable without a GPU, and so is this one.
"""

# navigate.py argparse defaults — the real robot's settings.
DEFAULT_NUM_SAMPLES = 8     # -n: action samples drawn from the diffusion head
DEFAULT_WAYPOINT = 2        # -w: "close waypoints exhibit straight line motion"
DEFAULT_RADIUS = 4          # -r: topomap nodes either side of the current one
DEFAULT_CLOSE_THRESHOLD = 3 # -t: temporal distance before localizing onward


class DriverConfig:
    """How the policy steers — `navigate.py`'s four flags, as a config block.

    These four numbers decide the action as much as the checkpoint does, and
    until P4's architecture pass they were reachable only by editing the
    constants above. That made them the only dials in the harness with neither
    a config section nor a column in the results table: every other one
    (`success_radius_m`, `timeout_slack`, the trail spacing) has both, and a
    table written under a changed value was indistinguishable from one written
    under the default.

    That matters most for the two phases that come next. P5 is allowed to
    "sanity-tune only what fidelity allows", so it will move these; P6's
    fairness protocol (plan §7) says every arm faces identical settings. The
    defaults here are unchanged and stay unchanged — plan decision E is to
    mirror the real LoCoBot and tune nothing — but `label()` is what lets a row
    *prove* that rather than assert it.
    """

    def __init__(self, num_samples=DEFAULT_NUM_SAMPLES, waypoint=DEFAULT_WAYPOINT,
                 radius=DEFAULT_RADIUS, close_threshold=DEFAULT_CLOSE_THRESHOLD):
        self.num_samples = int(num_samples)
        self.waypoint = int(waypoint)
        self.radius = int(radius)
        self.close_threshold = int(close_threshold)

        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.waypoint < 0:
            raise ValueError("waypoint index must not be negative")
        if self.radius < 0:
            raise ValueError("radius must not be negative")

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def as_dict(self):
        return {
            "num_samples": self.num_samples,
            "waypoint": self.waypoint,
            "radius": self.radius,
            "close_threshold": self.close_threshold,
        }

    def label(self):
        """The row's provenance, in navigate.py's own flag letters: `n8w2r4t3`."""
        return "n{}w{}r{}t{}".format(self.num_samples, self.waypoint,
                                     self.radius, self.close_threshold)

    def is_deployment_default(self):
        """True when these are the real robot's settings, untouched."""
        return self.as_dict() == type(self)().as_dict()

    def summary(self):
        return "{} samples · waypoint #{} · radius {} · close threshold {}{}".format(
            self.num_samples, self.waypoint, self.radius, self.close_threshold,
            "" if self.is_deployment_default() else "  (TUNED — not navigate.py's defaults)")
