"""
Rule-based safety filters that post-process the DQN's actions.

Analogous in spirit to hybrid RL+rules systems: the network proposes, a small
deterministic layer disposes. Three filters of increasing conservatism are
defined here, exposed by evaluate.py as --smooth, --strict, --conservative:

  SafetyFilter (--smooth)
      1. Lane-change cooldown: after any lane change, further lane changes are
         blocked for `cooldown` steps (replaced by IDLE), unless an emergency
         is detected. This suppresses jittery zig-zagging.
      2. Emergency braking: if a same-lane vehicle is closer ahead than
         `brake_dist`, FASTER and IDLE are overridden to SLOWER, and the
         cooldown is lifted so the network may escape sideways.

  StrictFilter (--strict)
      Adds pre-braking near slow traffic. Note it also retunes same_lane_y
      (0.12 -> 0.13) and brake_dist (0.09 -> 0.13), so it is NOT a pure
      superset of SafetyFilter; the smooth -> strict step changes two things.

  ConservativeFilter (--conservative)
      Adds a speed cap and restricts lane changes to emergencies. This is the
      shipped configuration.

Provenance, stated plainly because the report makes an issue of it: these
three filters were compared, and the conservative one chosen, using the
90000+i seed block. That block therefore acted as a SELECTION set for the
filter choice, not a held-out set. The unbiased number for the shipped
configuration is measured on the untouched 50000+i block. See the report.
"""

import numpy as np

LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4


class SafetyFilter:
    def __init__(self, cooldown=3, same_lane_y=0.12, brake_dist=0.09):
        self.cooldown = cooldown
        self.same_lane_y = same_lane_y
        self.brake_dist = brake_dist
        self.since_lane_change = cooldown

    def reset(self):
        self.since_lane_change = self.cooldown

    def _closest_ahead(self, obs_flat):
        m = obs_flat.reshape(5, 5)
        d = float("inf")
        for i in range(1, 5):
            presence, x, y, vx, vy = m[i]
            if presence > 0.5 and abs(y) < self.same_lane_y and x > 0:
                d = min(d, x)
        return d

    def filter(self, action, obs_flat):
        d = self._closest_ahead(obs_flat)
        emergency = d < self.brake_dist

        if emergency:
            # allow sideways escape; otherwise force braking
            if action in (LANE_LEFT, LANE_RIGHT):
                self.since_lane_change = 0
                return action
            return SLOWER

        if action in (LANE_LEFT, LANE_RIGHT):
            if self.since_lane_change < self.cooldown:
                self.since_lane_change += 1
                return IDLE
            self.since_lane_change = 0
            return action

        self.since_lane_change += 1
        return action


class StrictFilter(SafetyFilter):
    """More conservative variant: additionally converts FASTER to IDLE when a
    same-lane vehicle is within `prebrake_dist`. Trades a little speed for a
    materially lower crash rate."""

    def __init__(self, cooldown=3, same_lane_y=0.13, brake_dist=0.13,
                 prebrake_dist=0.22):
        super().__init__(cooldown=cooldown, same_lane_y=same_lane_y,
                         brake_dist=brake_dist)
        self.prebrake_dist = prebrake_dist

    def filter(self, action, obs_flat):
        d = self._closest_ahead(obs_flat)
        if d < self.prebrake_dist and action == FASTER:
            action = IDLE
        return super().filter(action, obs_flat)


class ConservativeFilter(StrictFilter):
    """StrictFilter plus a speed cap and emergency-only lane changes.

    Diagnosis behind it: the residual crashes of the strict hybrid did not
    come from speed (capping ego vx at 0.30 left the crash count unchanged)
    but from discretionary lane changes. Restricting lane changes to
    emergencies (a same-lane vehicle inside brake_dist) removes most of them.

    Provenance note: this diagnosis and the choice of this filter were made
    using the 90000+i seed block, which therefore acted as a selection set
    for the filter, not a held-out set. The unbiased number for this
    configuration is measured on the untouched 50000+i block; see the
    report."""

    def __init__(self, vcap=0.30):
        super().__init__()
        self.vcap = vcap

    def filter(self, action, obs_flat):
        d = self._closest_ahead(obs_flat)
        if obs_flat[3] > self.vcap and action == FASTER:
            action = IDLE
        if action in (LANE_LEFT, LANE_RIGHT) and d >= self.brake_dist:
            action = IDLE
        return super().filter(action, obs_flat)


class CappedStrictFilter(StrictFilter):
    """StrictFilter plus the speed cap, WITHOUT the lane-change restriction.

    This exists to isolate the speed cap: comparing it against StrictFilter
    (same rules, no cap) and against ConservativeFilter (cap plus the
    lane-change restriction) shows which of the two additions carries the
    improvement. Reported as the speed-cap probe in the paper.
    """

    def __init__(self, vcap=0.30):
        super().__init__()
        self.vcap = vcap

    def filter(self, action, obs_flat):
        if obs_flat[3] > self.vcap and action == FASTER:
            action = IDLE
        return super().filter(action, obs_flat)
