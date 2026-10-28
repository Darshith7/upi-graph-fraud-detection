"""Post-test descriptive error review. No training or threshold selection.

Run from project root: python src/analyze_errors.py
Requires numpy and prepare_graph.py beside this script; no Torch/XGBoost imports.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
from prepare_graph import TemporalGraphIndex


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def counts(labels, alerts):
    positive = labels == 1
    return dict(true_positives=int(np.count_nonzero(alerts & positive)),
                false_positives=int(np.count_nonzero(alerts & ~positive)),
                false_negatives=int(np.count_nonzero(~alerts & positive)),
                true_negatives=int(np.count_nonzero(~alerts & ~positive)),
                alerts=int(np.count_nonzero(alerts)))


def alert_masks(scores, ids, fraction, rule):
    n = len(scores)
    if scores.shape != ids.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid scores or ID alignment')
    k = math.floor(n * fraction)
    if not 1 <= k <= n:
        raise ValueError('Invalid alert budget')
    order = np.lexsort((ids, -scores.astype(np.float64)))
    ranked = np.zeros(n, dtype=bool)
    ranked[order[:k]] = True
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    cutoff, comparator = float(rule['cutoff']), rule['comparator']
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1 or comparator not in ('>', '>='):
        raise ValueError('Invalid frozen threshold')
    fixed = scores.astype(np.float64) >= cutoff if comparator == '>=' else scores.astype(np.float64) > cutoff
    return {'rank_budget': ranked, 'fixed_score_rule': fixed}, ranks


def overlap_counts(labels, a, b):
    return {group: {'rows': int(np.count_nonzero(mask)),
                     'laundering': int(np.count_nonzero(mask & (labels == 1))),
                     'non_laundering': int(np.count_nonzero(mask & (labels == 0)))}
            for group, mask in (('both_alert', a & b), ('xgboost_only_alert', a & ~b),
                                ('graphsage_only_alert', ~a & b), ('neither_alert', ~a & ~b))}


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows, fields):
    esc = lambda x: html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(f) + '</th>' for f in fields) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(row.get(f, '')) + '</td>' for f in fields) + '</tr>' for row in rows) + '</tbody></table></div>'


def run(args):
    output, pending = args.output, args.output.with_name(args.output.name + '.building')
    if output.exists() or pending.exists():
        raise ValueError(f'{output} or {pending} already exists; nothing overwritten')
    if not 1 <= args.examples <= 20:
        raise ValueError('examples must be between 1 and 20')
    schema = read_json(args.features / 'feature_schema.json')
    protocol = read_json(args.evaluation / 'evaluation_protocol.json')
    metrics = read_json(args.evaluation / 'test_metrics.json')
    if read_json(args.evaluation / 'evaluation_run.json')['status'] != 'complete':
        raise ValueError('Test evaluation is incomplete')
    schema_hash = sha256(args.features / 'feature_schema.json')
    if protocol['feature_schema_sha256'] != schema_hash:
        raise ValueError('Evaluation and feature schema do not match')
    load = lambda path: np.load(path, mmap_mode='r', allow_pickle=False)
    ids = load(args.features / 'test_transaction_ids.npy')
    np.testing.assert_array_equal(ids, load(args.evaluation / 'test_transaction_ids.npy'))
    labels = load(args.features / 'test_y.npy')
    edges = load(args.features / 'test_edges.npy')
    times = load(args.features / 'test_time_seconds.npy')
    X = load(args.features / 'test_X.npy')
    n = int(schema['rows_by_split']['test'])
    if (ids.shape != (n,) or labels.shape != (n,) or edges.shape != (n, 2)
            or times.shape != (n,) or X.shape != (n, schema['feature_count'])):
        raise ValueError('Invalid test array shapes')
    if (np.unique(ids).size != n or not np.isin(labels, [0, 1]).all()
            or np.count_nonzero(labels) != schema['laundering_by_split']['test']):
        raise ValueError('Invalid test IDs or labels')
    fraction = float(protocol['alert_fraction'])
    scores, masks, ranks, checks = {}, {}, {}, {}
    print('Checking saved predictions and reproducing recorded alert counts...', flush=True)
    for model in ('xgboost', 'graphsage'):
        scores[model] = load(args.evaluation / f'{model}_test_scores.npy')
        masks[model], ranks[model] = alert_masks(scores[model], ids, fraction, protocol['frozen_rules'][model])
        checks[model] = {}
        for policy, mask in masks[model].items():
            observed = counts(labels, mask)
            if any(observed[key] != metrics[model][policy][key] for key in observed):
                raise ValueError(f'{model}/{policy}: saved evaluation counts do not match')
            checks[model][policy] = observed
    overlaps = {policy: overlap_counts(labels, masks['xgboost'][policy], masks['graphsage'][policy])
                for policy in ('rank_budget', 'fixed_score_rule')}
    a, b, positive = masks['xgboost']['rank_budget'], masks['graphsage']['rank_budget'], labels == 1
    categories = {
        'xgboost_false_alert': a & ~positive,
        'graphsage_false_alert': b & ~positive,
        'xgboost_missed_laundering': ~a & positive,
        'graphsage_missed_laundering': ~b & positive,
        'laundering_caught_only_by_xgboost': a & ~b & positive,
        'laundering_caught_only_by_graphsage': ~a & b & positive,
        'laundering_missed_by_both': ~a & ~b & positive,
        'non_laundering_flagged_by_both': a & b & ~positive,
    }
    selected = {}
    samples = {}
    for number, (category, mask) in enumerate(categories.items()):
        candidates = np.flatnonzero(mask)
        # Stable sampling by transaction ID; each category uses its own deterministic seed.
        candidates = candidates[np.argsort(ids[candidates], kind='stable')]
        rng = np.random.default_rng(args.seed + number)
        rows = rng.choice(candidates, size=min(args.examples, len(candidates)), replace=False)
        rows = rows[np.argsort(ids[rows], kind='stable')]
        samples[category] = {'population_rows': len(candidates), 'transaction_ids': [int(ids[i]) for i in rows]}
        for i in rows:
            selected.setdefault(int(i), []).append(category)
    if not args.database.is_file():
        raise ValueError('Prepared database not found')
    print(f'Preparing {len(selected)} distinct cases and strictly earlier history...', flush=True)
    records, histories = [], []
    connection = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        db_manifest = json.loads(connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        if db_manifest['source_sha256'] != protocol['source_csv_sha256']:
            raise ValueError('Database comes from a different source CSV')
        with TemporalGraphIndex(args.graph) as graph:
            if graph.metadata['feature_schema_sha256'] != schema_hash:
                raise ValueError('Graph and features do not match')
            names = schema['feature_names']
            for i, groups in sorted(selected.items(), key=lambda item: int(ids[item[0]])):
                row = connection.execute('SELECT * FROM transactions WHERE transaction_id=?', (int(ids[i]),)).fetchone()
                if row is None or row['split'] != 'test' or row['label'] != int(labels[i]):
                    raise ValueError('Case transaction does not match test labels/split')
                origin = datetime.fromisoformat(schema['time_origin_in_source_timezone'])
                if int((datetime.fromisoformat(row['timestamp']) - origin).total_seconds()) != int(times[i]):
                    raise ValueError('Case transaction and feature timestamp do not match')
                record = dict(row)
                record.update(categories='; '.join(groups), history_cutoff=row['timestamp'])
                for model in scores:
                    record[f'{model}_score'] = float(scores[model][i])
                    record[f'{model}_rank'] = int(ranks[model][i])
                    for policy in ('rank_budget', 'fixed_score_rule'):
                        record[f'{model}_{policy}_alert'] = bool(masks[model][policy][i])
                h = graph.history_features(np.asarray(edges[i], dtype=np.int64), int(times[i]))
                for role_index, role in enumerate(('sender', 'receiver')):
                    start = names.index(role + '_log1p_prior_out_count')
                    np.testing.assert_allclose(h[role_index], X[i, start:start+6], rtol=1e-5, atol=1e-5)
                    for j, direction in enumerate(('outgoing', 'incoming')):
                        record[f'{role}_prior_{direction}_count_approx'] = int(round(math.expm1(float(h[role_index, j]))))
                        record[f'{role}_seconds_since_{direction}_approx'] = (
                            int(round(math.expm1(float(h[role_index, j+2])))) if h[role_index, j+4] else '')
                        # Match the saved GraphSAGE fanout and self-transfer policy.
                        prior = graph.recent_neighbors(int(edges[i, role_index]), int(times[i]),
                            limit=int(protocol['graphsage_fanout']), direction=direction, exclude_self=True)
                        for tx_id, tx_time in zip(prior['transaction_ids'], prior['times']):
                            if tx_time >= times[i]:
                                raise ValueError('History contains a current/future transaction')
                            prior_row = connection.execute('''SELECT transaction_id,timestamp,sender_bank,
                                sender_account,receiver_bank,receiver_account,amount_paid,payment_currency,
                                amount_received,receiving_currency,payment_format FROM transactions
                                WHERE transaction_id=?''', (int(tx_id),)).fetchone()
                            if prior_row is None or prior_row['timestamp'] >= row['timestamp']:
                                raise ValueError('Database history timestamp mismatch')
                            histories.append({'case_transaction_id': int(ids[i]), 'endpoint': role,
                                              'direction': direction, 'history_cutoff': row['timestamp'], **dict(prior_row)})
                records.append(record)
    finally:
        connection.close()
    summary = {
        'analysis_type': 'post-test descriptive error review; no fitting or threshold changes',
        'test_rows': n, 'seed': args.seed, 'examples_per_category': args.examples,
        'case_policy': 'offline rank_budget', 'alert_fraction': fraction,
        'primary_model_selected_before_test': protocol['primary_model'],
        'reproduced_counts': checks, 'overlap_by_policy': overlaps, 'samples': samples,
        'distinct_review_cases': len(records), 'history_context_rows': len(histories),
        'input_fingerprints': {str(p): sha256(p) for p in (
            args.evaluation / 'evaluation_protocol.json', args.evaluation / 'test_metrics.json',
            args.evaluation / 'xgboost_test_scores.npy', args.evaluation / 'graphsage_test_scores.npy')},
        'notes': [
            'False alerts and misses are defined against synthetic transaction labels, not verified account roles.',
            'Examples are uniformly sampled within each named category; they are not a representative sample of all transactions.',
            'A case may belong to several categories. Empty categories are reported explicitly.',
            'Raw scores from different models are not directly comparable calibrated probabilities. Ranks use each model separately.',
            'Displayed past transactions are input context, not evidence of causal feature importance or a laundering chain.',
            'Counts and recencies reconstructed from float32 logarithms are approximate; history contains transactions, not unique counterparties.',
            'No test-driven retuning. SHAP/GNNExplainer remain a later project step.',
        ],
    }
    pending.mkdir(parents=True)
    write_json(pending / 'error_summary.json', summary)
    fields = list(records[0]) if records else ['transaction_id', 'categories']
    write_csv(pending / 'review_cases.csv', records, fields)
    history_fields = list(histories[0]) if histories else ['case_transaction_id', 'endpoint', 'direction', 'transaction_id', 'timestamp']
    write_csv(pending / 'prior_transactions.csv', histories, history_fields)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Step 6 — Model error review</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;color:#172b3a;background:#f6f8fa}h1,h2,h3{line-height:1.2}section{background:white;border:1px solid #dbe2e8;border-radius:10px;padding:22px;margin:24px 0}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #dbe2e8;padding:8px;text-align:left;vertical-align:top}th{background:#eaf0f5}.scroll{overflow-x:auto}a{color:#14618c}.note{color:#4b5f6c}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}</style><body>',
             '<h1>Step 6: inspect model errors</h1><p>Saved predictions only. Models and thresholds remain frozen.</p>',
             '<p class="note">These are synthetic transaction labels. A false alert means a transaction labelled non-laundering was flagged. A miss means a transaction labelled laundering was not flagged. Account involvement does not establish a mule role.</p>']
    for policy, values in overlaps.items():
        parts += ['<section><h2>' + html.escape(policy.replace('_', ' ')) + ': agreement</h2>',
                  table([{'group': k, **v} for k, v in values.items()], ['group', 'rows', 'laundering', 'non_laundering']), '</section>']
    parts += ['<section><h2>Case selection</h2><p>Up to ' + str(args.examples) + ' cases per category, sampled with seed ' + str(args.seed) + '. All categories below use the offline top-' + f'{fraction:.2%}' + ' policy. Cases may appear in multiple categories.</p><ul>']
    for category, sample in samples.items():
        links = ', '.join(f'<a href="#case-{tx}">{tx}</a>' for tx in sample['transaction_ids']) or 'No cases in this category'
        parts.append(f'<li><b>{html.escape(category.replace("_", " "))}</b> ({sample["population_rows"]:,} total): {links}</li>')
    parts += ['</ul></section><p>For each case, compare the two ranks and decisions, inspect the payment details and prior activity, and record what remains unexplained. Context is not a model explanation; SHAP/GNNExplainer comes later.</p>']
    for record in records:
        tx = record['transaction_id']
        parts.append(f'<section id="case-{tx}"><h2>Transaction {tx}</h2><p>{html.escape(record["categories"])}</p>')
        parts.append(table([{'field': k, 'value': v} for k, v in record.items()], ['field', 'value']))
        evidence = [r for r in histories if r['case_transaction_id'] == tx]
        parts.append('<details><summary>Earlier transaction context (' + str(len(evidence)) + ' rows)</summary>')
        parts.append(table(evidence, history_fields) if evidence else '<p>No sampled prior non-self transfers for these endpoints.</p>')
        parts.append('</details></section>')
    parts += ['<section><h2>Interpretation limits</h2><ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in summary['notes']) + '</ul></section></body></html>']
    (pending / 'error_review.html').write_text('\n'.join(parts), encoding='utf-8')
    pending.rename(output)
    print('\nERROR ANALYSIS EXPORT COMPLETE')
    print('Saved counts reproduced for both policies and both models.')
    print('At the fixed ranking budget:')
    for name, value in overlaps['rank_budget'].items():
        print(f"  {name}: {value['rows']:,} transactions; {value['laundering']:,} laundering")
    print(f'Review cases: {len(records)}; prior-context rows: {len(histories)}')
    print(f'Open: {output / "error_review.html"}')
    print(f'Share for review: {output / "error_summary.json"} and {output / "review_cases.csv"}')
    print('No models loaded or retrained. Test thresholds unchanged. Tail excluded.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features'))
    parser.add_argument('--graph', type=Path, default=Path('data/temporal_graph'))
    parser.add_argument('--database', type=Path, default=Path('data/processed/transactions.sqlite'))
    parser.add_argument('--evaluation', type=Path, default=Path('reports/test_evaluation_v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/error_analysis_v1'))
    parser.add_argument('--examples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
