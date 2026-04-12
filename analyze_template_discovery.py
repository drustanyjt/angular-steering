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


def flags_for(text, strict=False):
    """
    Return (primary_flags, secondary_flags) sets for a generated text.

    strict=False (default, legacy): any 3-gram repeating >=4 times fires REP.
        Over-flags successful enthusiastic outputs like "How amazing! How
        wonderful! How incredible!" because the "How X!" pattern repeats.

    strict=True: REP requires either (a) a 3-gram repeating >=6 times, OR
        (b) a single token accounting for >=35% of all tokens (catches
        "I, I, I, I, I, ..." loops even when trigrams aren't uniform).
        Correctly separates pathology from strong sentiment-steered text.
    """
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
            rep_threshold = 6 if strict else 4
            if top_count >= rep_threshold:
                primary.add("REP")
        if strict:
            wc = Counter(words)
            top_word, top_n = wc.most_common(1)[0]
            if top_n >= 12 and top_n / len(words) > 0.35:
                primary.add("DOM")
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
    parser.add_argument("--strict", action="store_true",
                        help="use strict REP detection (3-gram >=6 repeats, "
                             "or single-token >=35%% dominance) — "
                             "avoids over-flagging enthusiastic adjective stacks")
    parser.add_argument("--flip-analysis", action="store_true",
                        help="also print 0°→180° sentiment-flip counts per template")
    args = parser.parse_args()

    stats = defaultdict(lambda: {
        "n": 0, "REP": 0, "CN": 0, "EMPTY": 0, "DOM": 0,
        "LOW_DIV": 0, "OFFTOPIC": 0, "any_primary": 0,
        "examples": [],
    })

    # For sentiment-flip analysis: outputs[(template, tweet_id)][angle] = text
    # and ground_truths[tweet_id] = 'positive' | 'negative' | 'neutral'
    flip_outputs = defaultdict(dict)
    ground_truths = {}

    templates_order = []
    angles_set = set()
    with open_csv(args.csv_path) as f:
        reader = csv.DictReader(f)
        schema = detect_schema(reader.fieldnames)
        has_gt = "ground_truth" in reader.fieldnames
        has_tid = "tweet_id" in reader.fieldnames
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
            prim, sec = flags_for(text, strict=args.strict)
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
            # Collect for flip analysis
            if args.flip_analysis and has_tid and label != "baseline":
                tid = row["tweet_id"]
                flip_outputs[(tname, tid)][label] = text
                if has_gt and tid not in ground_truths:
                    ground_truths[tid] = row["ground_truth"]

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

    if args.flip_analysis and flip_outputs:
        POS_WORDS = {'happy','great','wonderful','amazing','love','joy','excited',
                     'awesome','good','best','perfect','beautiful','glad','delighted',
                     'pleased','thrilled','fantastic','brilliant','cheerful','ecstatic',
                     'blessed','lucky','grateful','thankful','positive','nice'}
        NEG_WORDS = {'sad','bad','terrible','horrible','hate','angry','mad','upset',
                     'awful','worst','disgusted','annoyed','frustrated','depressed',
                     'furious','miserable','disappointed','gloomy','dreadful','negative',
                     'painful','sorry','regret','worry','exhausted','tired'}

        def senti_score(text):
            t = text.lower()
            return sum(1 for w in POS_WORDS if w in t) - sum(1 for w in NEG_WORDS if w in t)

        print(f"\n=== Sentiment flip analysis (0° → 180°) ===\n")
        print(f"{'template':<18}  {'n':>4}  {'sign_flips':>10}  "
              f"{'mean_Δscore':>12}  {'by GT (pos→→/neg→→)':>24}")
        for t in templates_order:
            shifts = []
            pos_shifts = []
            neg_shifts = []
            for (tpl, tid), angs in flip_outputs.items():
                if tpl != t:
                    continue
                if "0" in angs and "180" in angs:
                    s0 = senti_score(angs["0"])
                    s180 = senti_score(angs["180"])
                    d = s180 - s0
                    shifts.append(d)
                    if ground_truths.get(tid) == "positive":
                        pos_shifts.append(d)
                    elif ground_truths.get(tid) == "negative":
                        neg_shifts.append(d)
            if not shifts:
                continue
            flipped = sum(1 for d in shifts if d < 0) + sum(1 for d in shifts if d > 0 and False)
            # More useful: count sign-flips (either direction)
            sign_flips = 0
            for (tpl, tid), angs in flip_outputs.items():
                if tpl != t or "0" not in angs or "180" not in angs:
                    continue
                s0 = senti_score(angs["0"])
                s180 = senti_score(angs["180"])
                if (s0 > 0 and s180 < 0) or (s0 < 0 and s180 > 0):
                    sign_flips += 1
            mean_d = sum(shifts) / len(shifts)
            pos_mean = sum(pos_shifts) / max(len(pos_shifts), 1) if pos_shifts else 0
            neg_mean = sum(neg_shifts) / max(len(neg_shifts), 1) if neg_shifts else 0
            print(f"{t:<18}  {len(shifts):>4}  {sign_flips:>10}  "
                  f"{mean_d:>+11.2f}  pos:{pos_mean:+.2f}  neg:{neg_mean:+.2f}")

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
