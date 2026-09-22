"""Paired contrasts between arms, and the additivity check — task by task.

`p6_2_compare.py` puts each arm's mean beside the others'. This asks the
questions those columns cannot: by how much does one arm beat another *on the
same tasks*, and does a combined recipe gain what its parts gain, added up?
Both are paired, because every arm faced the same tasks under the same seeds —
and both take the task as the unit (`task_stats`): a task's seeds are averaged
first, and every SE and test is across tasks.

GPU-free; it reads tables the scoring runs wrote. Driven by a config's
`contrasts:` section, and refuses to run unless the tables pass the same
fairness gate as the comparison (`comparison.load_fair_tables`):

    contrasts:
      tables:                     # tables this run did not score, by label
        best_combined_p6: outputs/p6_1_metrics/best_combined.csv
      pairs:                      # each is `a - b`
        - [best_combined, clean_stock]
      additivity:                 # optional: a 2x2 around a baseline
        baseline: ctx03
        parts: [stride3, img160x120]
        combined: bc30
      restrict_to_task_set: false # optional: see `ContrastsConfig`

    ./sim_eval/run_p7_contrasts.sh --config sim_eval/configs/p7_2_e1.yaml

Writes beside the metrics, under the config's `comparison:` name:

    <name>_paired.md / .csv       each pair's Δ ± SE per statistic, paired t and
                                  Wilcoxon p, and McNemar on success when there
                                  is one seed per task; overall and per house
    <name>_paired_tasks.csv       every pair's per-task Δ — what those average
    <name>_additivity.md / .csv   Δ of each part, their sum, Δ of the
                                  combination, the interaction and a verdict
"""

import argparse
from pathlib import Path

import comparison
import metrics
import run_eval
import task_set
import task_stats
from comparison import ALL_HOUSES

STATISTICS = [statistic for statistic, _column, _over in metrics.EPISODE_STATISTICS]


class ContrastsConfig:
    """A config's `contrasts:` section, its arms resolved to tables on disk."""

    def __init__(self, table_paths, pairs, additivity=None, restrict_to_task_set=False):
        self.table_paths = table_paths
        self.pairs = [tuple(pair) for pair in pairs]
        self.additivity = additivity
        # Off by default. On, a table scored on a *larger* task set is cut down
        # to this config's (task, seed) episodes before the fairness gate — how
        # bc30 (P6's 2x10) is compared with best_combined (E2's 3x20) on the
        # trails they share. Only sound when those trails are the same trails,
        # which the config using it must be able to say (E2 adopted P6's).
        # Missing episodes are still refused.
        self.restrict_to_task_set = bool(restrict_to_task_set)

    @classmethod
    def from_yaml(cls, path, eval_config):
        section = run_eval.read_layered_yaml(path).get("contrasts") or {}
        table_paths = {name: eval_config.csv_path(name)
                       for name in eval_config.checkpoint_names}
        table_paths.update({label: run_eval.resolve_path(table)
                            for label, table in (section.get("tables") or {}).items()})
        config = cls(table_paths, section.get("pairs") or [], section.get("additivity"),
                     section.get("restrict_to_task_set", False))
        config.check_names()
        return config

    def names(self):
        """Every table a pair or the additivity check reads."""
        used = {name for pair in self.pairs for name in pair}
        if self.additivity:
            used |= {self.additivity["baseline"], self.additivity["combined"],
                     *self.additivity["parts"]}
        return used

    def check_names(self):
        unknown = sorted(self.names() - set(self.table_paths))
        if unknown:
            raise ValueError("contrasts name tables that are neither a checkpoint "
                             "of this run nor under `contrasts.tables`: {}"
                             .format(", ".join(unknown)))
        if self.additivity and len(self.additivity["parts"]) != 2:
            raise ValueError("additivity is a 2x2: exactly two `parts`")


def in_house(rows, house):
    return rows if house == ALL_HOUSES else [row for row in rows if row["scene"] == house]


def _fmt(value, places=3):
    return "n/a" if value is None else "{:+.{}f}".format(value, places)


def _p(value):
    return "n/a" if value is None else "{:.3f}".format(value)


def delta_cell(result):
    """`+0.050 ± 0.135 (20)`, flagged when it is beyond 2 SE of zero."""
    se = "n/a" if result.se is None else "{:.3f}".format(result.se)
    cell = "{} ± {} ({})".format(_fmt(result.mean), se, result.n)
    return cell if result.within_noise else "**{}**".format(cell)


# --- paired -------------------------------------------------------------------

def single_seed(tables):
    """One episode per task in every table — when McNemar is defined."""
    return all(set(task_stats.seed_counts(rows).values()) == {1}
               for rows in tables.values())


def pair_rows(a, b, tables, houses):
    """One CSV row per (scope, statistic) for the pair `a - b`."""
    rows = []
    mcnemar_ok = single_seed({a: tables[a], b: tables[b]})
    for house in [ALL_HOUSES] + houses:
        scoped = {a: in_house(tables[a], house), b: in_house(tables[b], house)}
        for statistic in STATISTICS:
            result = task_stats.contrast(scoped, {a: 1, b: -1}, statistic)
            row = {"pair": "{} - {}".format(a, b), "scope": house,
                   "statistic": statistic, "mean": comparison.round_cell(result.mean),
                   "se": comparison.round_cell(result.se), "n_tasks": result.n,
                   "within_noise": int(result.within_noise),
                   "p_paired_t": comparison.round_cell(result.p_paired_t),
                   "p_wilcoxon": comparison.round_cell(result.p_wilcoxon),
                   "mcnemar_a_only": "", "mcnemar_b_only": "", "mcnemar_p": ""}
            if statistic == "success_rate" and mcnemar_ok:
                a_only, b_only, p = task_stats.mcnemar(scoped[a], scoped[b])
                row.update(mcnemar_a_only=a_only, mcnemar_b_only=b_only,
                           mcnemar_p=comparison.round_cell(p))
            row["_result"] = result
            rows.append(row)
    return rows


PAIRED_COLUMNS = ("pair", "scope", "statistic", "mean", "se", "n_tasks",
                  "within_noise", "p_paired_t", "p_wilcoxon",
                  "mcnemar_a_only", "mcnemar_b_only", "mcnemar_p")


def pair_task_rows(a, b, tables):
    """One row per task: the pair's Δ on every statistic, for that task."""
    results = {statistic: task_stats.contrast(tables, {a: 1, b: -1}, statistic)
               for statistic in STATISTICS}
    scene_of = {row["task_id"]: row["scene"] for row in tables[a]}
    tasks = sorted(set(scene_of) & {row["task_id"] for row in tables[b]})
    return [dict({"pair": "{} - {}".format(a, b), "scene": scene_of[task],
                  "task_id": task},
                 **{statistic: comparison.round_cell(results[statistic].per_task.get(task))
                    for statistic in STATISTICS})
            for task in tasks]


def paired_markdown(title, pairs_rows, single):
    lines = ["# {} — paired contrasts".format(title), "",
             "Δ = a − b, per task (seeds averaged first), then mean ± SE across "
             "tasks (n). **Bold** = beyond 2 SE of zero.",
             "p: paired t and Wilcoxon signed-rank across tasks" +
             ("; McNemar (exact) on success, one seed per task." if single
              else ". No McNemar: several seeds per task."), ""]
    for pair, rows in pairs_rows:
        lines += ["## {}".format(pair), "", "### All houses", "",
                  "| statistic | Δ ± SE (n) | p paired t | p Wilcoxon |",
                  "|---|---|---|---|"]
        for row in rows:
            if row["scope"] == ALL_HOUSES:
                result = row["_result"]
                lines.append("| {} | {} | {} | {} |".format(
                    row["statistic"], delta_cell(result),
                    _p(result.p_paired_t), _p(result.p_wilcoxon)))
        success = [row for row in rows if row["statistic"] == "success_rate"]
        header = "| scope | success Δ ± SE (n) | p Wilcoxon |"
        rule = "|---|---|---|"
        if single:
            header += " McNemar a-only / b-only | McNemar p |"
            rule += "---|---|"
        lines += ["", "### Success by house", "", header, rule]
        for row in success:
            cells = [row["scope"], delta_cell(row["_result"]),
                     _p(row["_result"].p_wilcoxon)]
            if single:
                cells += ["{} / {}".format(row["mcnemar_a_only"], row["mcnemar_b_only"]),
                          _p(row["mcnemar_p"] if row["mcnemar_p"] != "" else None)]
            lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
        lines.append("")
    return "\n".join(lines)


# --- additivity ---------------------------------------------------------------

def additivity_rows(spec, tables):
    """Per statistic: Δ each part, their sum, Δ combined, and the interaction.

    Every quantity is a per-task contrast, so each SE is across tasks and
    already carries the pairing — the sum's SE is not the two SEs added.
    """
    base, (first, second), both = spec["baseline"], spec["parts"], spec["combined"]
    weights = {
        "first": {first: 1, base: -1},
        "second": {second: 1, base: -1},
        "sum": {first: 1, second: 1, base: -2},
        "both": {both: 1, base: -1},
        "interaction": {both: 1, first: -1, second: -1, base: 1},
    }
    rows = []
    for statistic in STATISTICS:
        results = {key: task_stats.contrast(tables, weight, statistic)
                   for key, weight in weights.items()}
        interaction = results["interaction"]
        row = {"statistic": statistic, "n_tasks": interaction.n,
               "verdict": task_stats.additivity_verdict(statistic, interaction),
               "p_interaction_t": comparison.round_cell(interaction.p_paired_t),
               "p_interaction_wilcoxon": comparison.round_cell(interaction.p_wilcoxon)}
        for key, result in results.items():
            row[key] = comparison.round_cell(result.mean)
            row[key + "_se"] = comparison.round_cell(result.se)
        row["_results"] = results
        rows.append(row)
    return rows


ADDITIVITY_COLUMNS = ("statistic", "n_tasks", "first", "first_se", "second",
                      "second_se", "sum", "sum_se", "both", "both_se",
                      "interaction", "interaction_se", "p_interaction_t",
                      "p_interaction_wilcoxon", "verdict")


def additivity_markdown(title, spec, rows):
    base, (first, second), both = spec["baseline"], spec["parts"], spec["combined"]
    lines = [
        "# {} — additivity".format(title), "",
        "Every Δ is against **{}**, per task (seeds averaged first), then mean ± SE "
        "across tasks (n). **Bold** = beyond 2 SE of zero.".format(base),
        "Interaction = Δ{b} − (Δ{f} + Δ{s}) = {b} − {f} − {s} + {base}, per task.".format(
            b=both, f=first, s=second, base=base),
        "Verdict: *additive* when the interaction is within 2 SE; beyond it, "
        "*super-additive* = the combination does better than the sum predicts, in "
        "the statistic's better direction, *sub-additive* = worse. path_length_m "
        "has no better direction.", "",
        "| statistic | Δ{f} | Δ{s} | Δ{f} + Δ{s} | Δ{b} | interaction | p (t / Wilcoxon) | verdict |"
        .format(f=first, s=second, b=both),
        "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        results = row["_results"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} / {} | {} |".format(
            row["statistic"], *(delta_cell(results[key]) for key in
                                ("first", "second", "sum", "both", "interaction")),
            _p(results["interaction"].p_paired_t), _p(results["interaction"].p_wilcoxon),
            row["verdict"]))
    return "\n".join(lines) + "\n"


# --- the run ------------------------------------------------------------------

def strip_private(rows):
    return [{key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows]


def write_paired(contrasts, tables, naming, output):
    """The paired tables for every configured pair. Returns the paths written."""
    houses = comparison.houses(tables)
    pairs_rows = [("{} - {}".format(a, b), pair_rows(a, b, tables, houses))
                  for a, b in contrasts.pairs]
    markdown = output / naming.output("paired.md")
    table = output / naming.output("paired.csv")
    per_task = output / naming.output("paired_tasks.csv")
    comparison.write_csv(table, strip_private(row for _pair, rows in pairs_rows
                                              for row in rows), PAIRED_COLUMNS)
    markdown.write_text(paired_markdown(naming.title, pairs_rows, single_seed(tables)))
    comparison.write_csv(per_task, [row for a, b in contrasts.pairs
                                    for row in pair_task_rows(a, b, tables)],
                         ("pair", "scene", "task_id") + tuple(STATISTICS))
    return [markdown, table, per_task]


def write_additivity(contrasts, tables, naming, output):
    """The additivity table for the configured 2x2. Returns the paths written."""
    rows = additivity_rows(contrasts.additivity, tables)
    markdown = output / naming.output("additivity.md")
    table = output / naming.output("additivity.csv")
    comparison.write_csv(table, strip_private(rows), ADDITIVITY_COLUMNS)
    markdown.write_text(additivity_markdown(naming.title, contrasts.additivity, rows))
    return [markdown, table]


def run(config_path):
    """Write every file the config asks for. Returns the paths written."""
    eval_config = run_eval.EvalConfig.from_yaml(config_path)
    naming = comparison.ComparisonNaming.from_yaml(config_path)
    contrasts = ContrastsConfig.from_yaml(config_path, eval_config)
    tables = comparison.load_fair_tables(
        eval_config, {name: contrasts.table_paths[name]
                      for name in sorted(contrasts.names())},
        restrict_to_task_set=contrasts.restrict_to_task_set)
    written = []
    if contrasts.pairs:
        written += write_paired(contrasts, tables, naming, eval_config.output_dir)
    if contrasts.additivity:
        written += write_additivity(contrasts, tables, naming, eval_config.output_dir)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="a run config with a `contrasts:` section")
    args = parser.parse_args()
    for path in run(args.config):
        print("wrote:      {}".format(path))


if __name__ == "__main__":
    try:
        main()
    except (comparison.ComparisonError, task_set.TaskSetError, ValueError,
            FileNotFoundError) as error:
        raise SystemExit("FAILED: {}".format(error))
