#!/usr/bin/env python3
"""
Extract validation and test metrics from SHOT training log files.
Generates a CSV with paired metrics at each evaluation epoch for correlation analysis.
"""

import re
import argparse
import csv
from pathlib import Path


def parse_log_file(log_path: str) -> list[dict]:
    """Parse a SHOT training log file and extract metrics at each periodic evaluation."""

    with open(log_path, 'r') as f:
        content = f.read()

    results = []

    # Split by periodic evaluation blocks
    # Pattern: "--- Periodic Evaluation at Epoch X ---"
    eval_pattern = r'--- Periodic Evaluation at Epoch (\d+) ---'
    eval_matches = list(re.finditer(eval_pattern, content))

    for i, match in enumerate(eval_matches):
        epoch = int(match.group(1))

        # Get the text block for this evaluation (until next eval or end)
        start_pos = match.end()
        if i + 1 < len(eval_matches):
            end_pos = eval_matches[i + 1].start()
        else:
            # For the last eval block, go until "Restored best checkpoint" or end
            restored_match = re.search(r'Restored best checkpoint', content[start_pos:])
            if restored_match:
                end_pos = start_pos + restored_match.start()
            else:
                end_pos = len(content)

        block = content[start_pos:end_pos]

        # Extract test metrics
        metrics = {'epoch': epoch}

        # Transfer accuracy (testing on new modality only)
        transfer_match = re.search(r'Transfer accuracy: ([\d.]+)%', block)
        if transfer_match:
            metrics['test_transfer'] = float(transfer_match.group(1))

        # Peeking accuracy (testing on starting modality only)
        peeking_match = re.search(r'Peeking accuracy: ([\d.]+)%', block)
        if peeking_match:
            metrics['test_peeking'] = float(peeking_match.group(1))

        # Addition accuracy (testing on both modalities)
        addition_match = re.search(r'Addition accuracy: ([\d.]+)%', block)
        if addition_match:
            metrics['test_addition'] = float(addition_match.group(1))

        # Peeking+transfer ensemble (only available in addition objective)
        ensemble_match = re.search(r'Test Accuracy \(peeking\+transfer ensemble\): ([\d.]+)%', block)
        if ensemble_match:
            metrics['test_ensemble'] = float(ensemble_match.group(1))

        # Validation metrics
        val_peeking_match = re.search(r'Val peeking accuracy: ([\d.]+)%', block)
        if val_peeking_match:
            metrics['val_peeking'] = float(val_peeking_match.group(1))

        val_agreement_match = re.search(r'Val teacher agreement: ([\d.]+)%', block)
        if val_agreement_match:
            metrics['val_teacher_agreement'] = float(val_agreement_match.group(1))

        # Check if this was a new best checkpoint
        best_match = re.search(r'>> New best checkpoint \(val combined: ([\d.]+)\)', block)
        metrics['is_best'] = best_match is not None
        if best_match:
            metrics['val_combined'] = float(best_match.group(1))
        else:
            # Compute val_combined if we have the components (default weights: 0.5 each)
            if 'val_peeking' in metrics and 'val_teacher_agreement' in metrics:
                metrics['val_combined'] = 0.5 * metrics['val_peeking'] + 0.5 * metrics['val_teacher_agreement']

        results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from SHOT training logs')
    parser.add_argument('log_files', nargs='+', help='Log file(s) to parse')
    parser.add_argument('-o', '--output', default='val_test_metrics.csv', help='Output CSV file')
    parser.add_argument('--include-source', action='store_true', help='Include source file column')
    args = parser.parse_args()

    all_results = []

    for log_file in args.log_files:
        print(f"Parsing: {log_file}")
        results = parse_log_file(log_file)

        if args.include_source:
            for r in results:
                r['source_file'] = Path(log_file).name

        all_results.extend(results)
        print(f"  Found {len(results)} evaluation points")

    if not all_results:
        print("No evaluation points found!")
        return

    # Determine columns (use first result as template, but ensure consistent ordering)
    base_columns = ['epoch', 'val_peeking', 'val_teacher_agreement', 'val_combined',
                    'test_peeking', 'test_transfer', 'test_addition', 'test_ensemble', 'is_best']
    if args.include_source:
        base_columns.insert(0, 'source_file')

    # Filter to only columns that exist in the data
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    columns = [c for c in base_columns if c in all_keys]

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nWritten {len(all_results)} rows to {args.output}")
    print(f"Columns: {', '.join(columns)}")


if __name__ == '__main__':
    main()
