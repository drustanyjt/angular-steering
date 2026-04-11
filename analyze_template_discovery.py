"""
Score a sentiment-steering CSV for REP/CN/EMPTY degeneration.

Reads any CSV produced by run_template_discovery.py or
run_sentiment_experiment.py (canonical schema) and prints a heatmap of
primary-flag rates per (template × angle), plus a per-template summary
of clean angle range.

Also accepts legacy discovery CSVs that used
`template,angle,text` columns instead of
`prompt_template,param_value,original_text` — it auto-detects the schema.

Primary metric: REP + CN + EMPTY.
Secondary (informational only): LOW_DIV, OFFTOPIC.

Usage:
    python analyze_template_discovery.py results/template_discovery_7B_round1_mode0.csv
    python analyze_template_discovery.py results/results_7B_20260412_15deg_restate_echo_en_similar_tweet_en_rewrite.csv
"""

import argparse
import csv
import gzip
import re
from collections import Counter, defaultdict


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"\w+")


def flags_for(text):
    """Return (primary_flags, secondary_flags) sets for a generated text."""
    primary = set()
    secondary = set()
    if len(text.strip()) < 10:
        primary.add("EMPTY")
    if CHINESE_RE.search(text):
        primary.add("CN")
    words = WORD_RE.findall(text.lower())
    if len(words) >= 9:
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        if trigrams:
            top_count = Counter(trigrams).most_common(1)[0][1]
            if top_count >= 4:
                primary.add("REP")
        uniq_ratio = len(set(words)) / len(words)
        if uniq_ratio < 0.3:
            secondary.add("LOW_DIV")
    return primary, secondary


def jaccard_flag(text, input_text):
    """OFFTOPIC if Jaccard overlap < 0.15 between non-trivial tokens."""
    STOP = {"a", "an", "the", "is", "it", "i", "to", "of", "in", "on", "and",
            "this", "that", "for", "with", "at", "as", "be", "or", "but",
            "my", "you", "your", "me", "we", "are", "was", "were", "s", "t"}
    gen_tokens = {w for w in WORD_RE.findall(text.lower()) if w not in STOP and len(w) > 2}
    in_tokens = {w for w in WORD_RE.findall(input_text.lower()) if w not in STOP and len(w) > 2}
    if not in_tokens:
        return False
    union = gen_tokens | in_tokens
    if not union:
        return True
    jaccard = len(gen_tokens & in_tokens) / len(union)
    return jaccard < 0.15


def open_csv(path):
    """Open a CSV or CSV.gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def detect_schema(fieldnames):
    """
    Return a dict mapping canonical field → actual column name.
    Supports:
      - canonical:  prompt_template, param_value, method, original_text
      - legacy discovery: template, angle, text (no method column)
    """
    fn = set(fieldnames)
    if {"prompt_template", "param_value", "method", "original_text"}.issubset(fn):
        return {
            "template": "prompt_template",
            "label": "param_value",
            "method": "method",
            "text": "original_text",
            "kind": "canonical",
        }
    if {"template", "angle", "text"}.issubset(fn):
        return {
            "template": "template",
            "label": "angle",
            "method": None,
            "text": "text",
            "kind": "legacy",
        }
    raise ValueError(
        f"Unrecognized CSV schema. Columns: {sorted(fn)}. Expected either canonical "
        "(prompt_template, param_value, method, original_text) or legacy "
        "(template, angle, text)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="percent threshold for 'clean' (default 10)")
    parser.add_argument("--show-examples", action="store_true",
                        help="print flagged example outputs per template")
    parser.add_argument("--no-jaccard", action="store_true",
                        help="skip Jaccard OFFTOPIC computation (faster)")
    args = parser.parse_args()

    stats = defaultdict(lambda: {
        "n": 0, "REP": 0, "CN": 0, "EMPTY": 0,
        "LOW_DIV": 0, "OFFTOPIC": 0, "any_primary": 0,
        "examples": [],
    })

    templates_order = []
    angles_set = set()
    with open_csv(args.csv_path) as f:
        reader = csv.DictReader(f)
        schema = detect_schema(reader.fieldnames)
        # In canonical mode we only want rows from angular / baseline. For
        # angular, label = the angle; baseline rows are bucketed as "baseline".
        for row in reader:
            if schema["kind"] == "canonical":
                method = row[schema["method"]]
                if method not in ("angular", "baseline"):
                    continue
                label = "baseline" if method == "baseline" else row[schema["label"]]
            else:
                label = row[schema["label"]]
            tname = row[schema["template"]]
            if tname not in templates_order:
                templates_order.append(tname)
            angles_set.add(label)
            text = row["generated_text"]
            prim, sec = flags_for(text)
            if not args.no_jaccard and jaccard_flag(text, row[schema["text"]]):
                sec.add("OFFTOPIC")
            key = (tname, label)
            s = stats[key]
            s["n"] += 1
            for flag in prim:
                s[flag] += 1
            for flag in sec:
                s[flag] += 1
            if prim:
                s["any_primary"] += 1
                if len(s["examples"]) < 3:
                    s["examples"].append((sorted(prim), text.strip()[:120]))

    # Sort angles (baseline first, then numeric)
    def angle_sort_key(a):
        if a == "baseline":
            return (0, 0)
        try:
            return (1, int(a))
        except ValueError:
            return (2, a)

    angles_sorted = sorted(angles_set, key=angle_sort_key)

    # Heatmap
    print(f"\n=== Primary-flag rate % (REP + CN + EMPTY) ===")
    print(f"    ({args.csv_path})\n")
    header = f"{'template':<20}  " + "  ".join(f"{a:>5}" for a in angles_sorted)
    print(header)
    print("-" * len(header))
    for t in templates_order:
        cells = []
        for a in angles_sorted:
            s = stats[(t, a)]
            if s["n"] == 0:
                cells.append("    -")
                continue
            pct = 100 * s["any_primary"] / s["n"]
            cells.append(f"{pct:5.1f}")
        print(f"{t:<20}  " + "  ".join(cells))

    # Per-flag breakdown
    print(f"\n=== Per-flag rate % ===\n")
    for flag in ["REP", "CN", "EMPTY"]:
        print(f"-- {flag} --")
        print(header)
        print("-" * len(header))
        for t in templates_order:
            cells = []
            for a in angles_sorted:
                s = stats[(t, a)]
                if s["n"] == 0:
                    cells.append("    -")
                    continue
                pct = 100 * s[flag] / s["n"]
                cells.append(f"{pct:5.1f}")
            print(f"{t:<20}  " + "  ".join(cells))
        print()

    # Template summary: widest circular contiguous clean run
    print(f"=== Template summary (threshold < {args.threshold}%) ===\n")
    print(f"{'template':<20}  {'clean_range':>12}  {'worst_ang':>10}  "
          f"{'worst_pct':>10}  {'baseline':>10}  {'mean_pct':>10}")

    numeric_angles = [a for a in angles_sorted if a != "baseline"]
    try:
        numeric_vals = sorted(int(a) for a in numeric_angles)
    except ValueError:
        numeric_vals = []

    summary_rows = []
    for t in templates_order:
        base = stats.get((t, "baseline"))
        base_pct = 100 * base["any_primary"] / base["n"] if base and base["n"] else float("nan")

        pcts = {}
        for a in numeric_angles:
            s = stats[(t, a)]
            if s["n"] == 0:
                continue
            pcts[int(a)] = 100 * s["any_primary"] / s["n"]
        if not pcts:
            continue

        step = numeric_vals[1] - numeric_vals[0] if len(numeric_vals) > 1 else 15
        clean_mask = [pcts.get(a, 100) < args.threshold for a in numeric_vals]
        doubled = clean_mask + clean_mask
        best_len = 0
        cur = 0
        for v in doubled:
            cur = cur + 1 if v else 0
            best_len = max(best_len, cur)
        best_len = min(best_len, len(clean_mask))
        clean_degrees = best_len * step

        worst_ang = max(pcts, key=lambda a: pcts[a])
        worst_pct = pcts[worst_ang]
        mean_pct = sum(pcts.values()) / len(pcts)
        summary_rows.append((t, clean_degrees, worst_ang, worst_pct, base_pct, mean_pct))

    summary_rows.sort(key=lambda r: (-r[1], r[5]))
    for t, clean, wa, wp, bp, mp in summary_rows:
        print(f"{t:<20}  {clean:>10}°  {wa:>10}  {wp:>9.1f}%  "
              f"{bp:>9.1f}%  {mp:>9.1f}%")

    if args.show_examples:
        print(f"\n=== Flagged examples (first per angle, per template) ===\n")
        for t in templates_order:
            print(f"-- {t} --")
            for a in angles_sorted:
                s = stats[(t, a)]
                if s["examples"]:
                    flags, ex = s["examples"][0]
                    print(f"  {a:>5}  {','.join(flags):<10}  {ex}")
            print()

    # Save scores CSV next to input (stripping .gz suffix if present)
    base = args.csv_path[:-3] if args.csv_path.endswith(".gz") else args.csv_path
    scores_path = base.replace(".csv", "_scores.csv")
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["template", "angle", "n", "any_primary_pct",
                         "REP_pct", "CN_pct", "EMPTY_pct",
                         "LOW_DIV_pct", "OFFTOPIC_pct"])
        for t in templates_order:
            for a in angles_sorted:
                s = stats[(t, a)]
                if s["n"] == 0:
                    continue
                writer.writerow([
                    t, a, s["n"],
                    f"{100*s['any_primary']/s['n']:.2f}",
                    f"{100*s['REP']/s['n']:.2f}",
                    f"{100*s['CN']/s['n']:.2f}",
                    f"{100*s['EMPTY']/s['n']:.2f}",
                    f"{100*s['LOW_DIV']/s['n']:.2f}",
                    f"{100*s['OFFTOPIC']/s['n']:.2f}",
                ])
    print(f"\nSaved scores to {scores_path}")


if __name__ == "__main__":
    main()
