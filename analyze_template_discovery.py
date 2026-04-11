"""
Score template discovery runs for REP/CN/EMPTY degeneration.

Reads a CSV produced by run_template_discovery.py and prints a heatmap of
primary-flag rates (REP + CN + EMPTY) per (template x angle), plus a
per-template summary of clean angle range. Primary metric is the one we
optimize for; secondary flags (LOW_DIV, OFFTOPIC) are reported informally.

Usage:
    python analyze_template_discovery.py results/template_discovery_7B_round1_mode0.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"\w+")


def flags_for(text):
    """Return (primary_flags, secondary_flags) sets for a generated text."""
    primary = set()
    secondary = set()
    stripped = text.strip()
    if len(stripped) < 10:
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
    """OFFTOPIC if jaccard overlap < 0.15 between non-trivial tokens."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="percent threshold for 'clean' (default 10)")
    parser.add_argument("--show-examples", action="store_true",
                        help="print up to 3 flagged example outputs per template")
    args = parser.parse_args()

    # counts[(template, angle)] = dict with fields
    stats = defaultdict(lambda: {
        "n": 0, "REP": 0, "CN": 0, "EMPTY": 0,
        "LOW_DIV": 0, "OFFTOPIC": 0, "any_primary": 0,
        "examples": [],
    })

    templates_order = []
    angles_set = set()
    with open(args.csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tname = row["template"]
            angle = row["angle"]
            if tname not in templates_order:
                templates_order.append(tname)
            angles_set.add(angle)
            text = row["generated_text"]
            prim, sec = flags_for(text)
            if jaccard_flag(text, row["text"]):
                sec.add("OFFTOPIC")
            key = (tname, angle)
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
                cells.append("   -")
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
                    cells.append("   -")
                    continue
                pct = 100 * s[flag] / s["n"]
                cells.append(f"{pct:5.1f}")
            print(f"{t:<20}  " + "  ".join(cells))
        print()

    # Template summary: find widest clean contiguous angle range
    print(f"=== Template summary (threshold < {args.threshold}%) ===\n")
    print(f"{'template':<20}  {'clean_range':>12}  {'worst_ang':>10}  "
          f"{'worst_pct':>10}  {'baseline':>10}  {'mean_pct':>10}")

    # Only angles (not baseline)
    numeric_angles = [a for a in angles_sorted if a != "baseline"]
    try:
        numeric_vals = sorted(int(a) for a in numeric_angles)
    except ValueError:
        numeric_vals = []

    summary_rows = []
    for t in templates_order:
        # baseline
        base = stats.get((t, "baseline"))
        base_pct = 100 * base["any_primary"] / base["n"] if base and base["n"] else float("nan")

        # Get pct per angle
        pcts = {}
        for a in numeric_angles:
            s = stats[(t, a)]
            if s["n"] == 0:
                continue
            pcts[int(a)] = 100 * s["any_primary"] / s["n"]
        if not pcts:
            continue

        # Widest circular contiguous run of angles under threshold
        step = numeric_vals[1] - numeric_vals[0] if len(numeric_vals) > 1 else 15
        clean_mask = [pcts.get(a, 100) < args.threshold for a in numeric_vals]
        # Double the mask for circular scanning
        doubled = clean_mask + clean_mask
        best_len = 0
        cur = 0
        for v in doubled:
            cur = cur + 1 if v else 0
            best_len = max(best_len, cur)
        best_len = min(best_len, len(clean_mask))  # cap at full circle
        clean_degrees = best_len * step

        worst_ang = max(pcts, key=lambda a: pcts[a])
        worst_pct = pcts[worst_ang]
        mean_pct = sum(pcts.values()) / len(pcts)
        summary_rows.append((t, clean_degrees, worst_ang, worst_pct, base_pct, mean_pct))

    # Sort by clean_range desc
    summary_rows.sort(key=lambda r: (-r[1], r[5]))
    for t, clean, wa, wp, bp, mp in summary_rows:
        print(f"{t:<20}  {clean:>10}°  {wa:>10}  {wp:>9.1f}%  "
              f"{bp:>9.1f}%  {mp:>9.1f}%")

    if args.show_examples:
        print(f"\n=== Flagged examples (first 3 per template) ===\n")
        for t in templates_order:
            print(f"-- {t} --")
            for a in angles_sorted:
                s = stats[(t, a)]
                if s["examples"]:
                    for flags, ex in s["examples"][:1]:
                        print(f"  {a:>5}  {','.join(flags):<10}  {ex}")
            print()

    # Save scores CSV
    scores_path = args.csv_path.replace(".csv", "_scores.csv")
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
