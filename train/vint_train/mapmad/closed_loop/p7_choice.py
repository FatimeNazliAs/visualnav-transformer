"""Phase 7 choice rule (p7_rules.md §6, block item 7 + Naz 3b), applied mechanically. Pure Python.

    winner, steps = choose(success, collisions, lens, height, training_like=("fish025", "fish055"))

success[c] = mean primary success of B_c (108 episodes x 3 seeds); collisions[c] = mean collisions per B_c run.
1. Best = highest success.
2. Training-like T = the better of the training_like cameras (tie -> the first listed); S(T) >= S(Best) - 0.05 -> T.
3. Else candidates = cameras with S >= S(Best) - 0.03; one candidate -> Best; else fewest collisions per run.
4. Still tied (collisions within 5% of the lowest, relative) -> fisheye over pinhole, then height closer to an indoor
   estimate, then higher success, then name.
"""

from typing import Dict, List, Sequence, Tuple

TRAINING_MARGIN = 0.05
NEAR_BEST = 0.03
COLLISION_TIE = 0.05  # relative
INDOOR_HEIGHTS = (0.25, 0.55)


def choose(success: Dict[str, float], collisions: Dict[str, float], lens: Dict[str, str], height: Dict[str, float],
           training_like: Sequence[str] = ("fish025", "fish055")) -> Tuple[str, List[str]]:
    """The winning camera and the rule steps taken (plain-text lines for choice.md)."""
    steps: List[str] = []
    best = max(sorted(success), key=lambda c: success[c])
    steps.append(f"1. Best = {best} (S = {success[best]:.4f})")
    t = max(training_like, key=lambda c: (success[c], -list(training_like).index(c)))
    steps.append(f"2. training-like T = {t} (S = {success[t]:.4f}); needs >= S(Best) - {TRAINING_MARGIN} = "
                 f"{success[best] - TRAINING_MARGIN:.4f}")
    if success[t] >= success[best] - TRAINING_MARGIN - 1e-12:
        steps.append(f"   -> T wins: {t}")
        return t, steps
    cands = sorted(c for c in success if success[c] >= success[best] - NEAR_BEST - 1e-12)
    steps.append(f"3. not met; candidates within {NEAR_BEST} of Best: {cands}")
    if cands == [best]:
        steps.append(f"   -> Best wins: {best}")
        return best, steps
    low = min(collisions[c] for c in cands)
    tied = [c for c in cands if collisions[c] <= low * (1.0 + COLLISION_TIE) + 1e-12]
    steps.append("   collisions per run: " + ", ".join(f"{c} {collisions[c]:.2f}" for c in cands))
    if len(tied) == 1:
        steps.append(f"   -> fewest collisions: {tied[0]}")
        return tied[0], steps

    def key(c: str):
        return (lens[c] != "fisheye", min(abs(height[c] - h) for h in INDOOR_HEIGHTS), -success[c], c)

    win = min(tied, key=key)
    steps.append(f"4. collisions within {int(COLLISION_TIE * 100)}%: {tied} -> fisheye, height near an indoor estimate, "
                 f"success, name -> {win}")
    return win, steps
