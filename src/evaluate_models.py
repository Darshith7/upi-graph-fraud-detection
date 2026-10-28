"""Evaluate frozen XGBoost/GraphSAGE models on the held-out main test period.

Run: python src/evaluate_models.py
Keep graphsage_trial.py and prepare_graph.py in src/.
No model fitting, threshold tuning, or tail evaluation is performed.
XGBoost and PyTorch run in separate fresh processes to avoid shared-runtime conflicts.
"""
import argparse
import csv
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

# Applied before native libraries are imported, including in fresh worker processes.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
faulthandler.enable(all_threads=True)

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def classification_counts(labels, alerts):
    positive = labels == 1
    tp = int(np.count_nonzero(alerts & positive))
    fp = int(np.count_nonzero(alerts & ~positive))
    fn = int(np.count_nonzero(~alerts & positive))
    tn = int(np.count_nonzero(~alerts & ~positive))
    return {'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
            'true_negatives': tn, 'alerts': tp + fp, 'alert_fraction': (tp + fp) / len(labels),
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'false_positives_per_1000_non_laundering': 1000 * fp / (fp + tn) if fp + tn else 0.0}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(folder, split, schema):
    # Labels deliberately excluded from inference inputs.
    data = {key: np.load(folder / f'{split}_{key}.npy', mmap_mode='r', allow_pickle=False)
            for key in ('X', 'edges', 'time_seconds', 'transaction_ids')}
    n = int(schema['rows_by_split'][split])
    expected = {'X': (n, schema['feature_count']), 'edges': (n, 2),
                'time_seconds': (n,), 'transaction_ids': (n,)}
    for key, value in data.items():
        if value.shape != expected[key]:
            raise ValueError(f'{split}/{key}: unexpected shape')
        if key != 'X' and value.dtype.kind not in 'iu':
            raise ValueError(f'{split}/{key}: expected integer array')
    if data['X'].dtype != np.float32 or n == 0:
        raise ValueError(f'{split}: expected nonempty float32 features')
    if np.any(data['time_seconds'][1:] < data['time_seconds'][:-1]):
        raise ValueError(f'{split}: unordered timestamps')
    if np.unique(data['transaction_ids']).size != n:
        raise ValueError(f'{split}: duplicate transaction IDs')
    if np.any(data['edges'] < 0) or np.any(data['edges'] >= schema['account_count']):
        raise ValueError(f'{split}: invalid account index')
    return data


def test_metrics(labels, scores, ids, fraction, rule):
    """Offline ranking and frozen deployment rule, with no threshold selection."""
    from sklearn.metrics import average_precision_score, auc, precision_recall_curve
    labels, scores, ids = map(np.asarray, (labels, scores, ids))
    if labels.ndim != 1 or labels.shape != scores.shape or ids.shape != scores.shape:
        raise ValueError('Unaligned labels, scores or IDs')
    if not np.isin(labels, [0, 1]).all() or not 0 < np.count_nonzero(labels) < len(labels):
        raise ValueError('Evaluation requires both binary classes')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid model scores')
    k = math.floor(len(labels) * fraction)
    if not 1 <= k <= len(labels):
        raise ValueError('Invalid ranking budget')
    ranking_alerts = np.zeros(len(labels), dtype=bool)
    ranking_alerts[np.lexsort((ids, -scores.astype(np.float64)))[:k]] = True
    cutoff, comparator = rule['cutoff'], rule['comparator']
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1 or comparator not in ('>', '>='):
        raise ValueError('Invalid frozen score rule')
    # float64 comparison preserves the exact saved cutoff against float32 scores.
    fixed_alerts = scores.astype(np.float64) >= cutoff if comparator == '>=' else scores.astype(np.float64) > cutoff
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        'evaluation_split': 'test', 'rows': len(labels),
        'laundering_rows': int(np.count_nonzero(labels)),
        'average_precision': float(average_precision_score(labels, scores)),
        'pr_auc_trapezoidal': float(auc(recall, precision)),
        'constant_score_average_precision': float(np.mean(labels)),
        'rank_budget': {'requested_alert_fraction': fraction, 'budget_rows': k,
                        'tie_break': 'descending score, ascending transaction ID',
                        **classification_counts(labels, ranking_alerts)},
        'fixed_score_rule': {**rule, 'selected_on': 'validation',
                             **classification_counts(labels, fixed_alerts)},
    }


def xgb_scores(booster, X, names, threads):
    import xgboost as xgb
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite features')
    matrix = xgb.DMatrix(X, feature_names=names, nthread=threads)
    return booster.predict(matrix)


def worker(args):
    """Each child imports exactly one model framework. Never import train_baseline here."""
    name, phase = args.worker, args.phase
    print(f'[{name}/{phase}] process started; CPU threads=1', flush=True)
    schema = read_json(args.features / 'feature_schema.json')
    folder = args.xgboost if name == 'xgboost' else args.graphsage
    split = 'validation' if phase == 'check' else 'test'
    metadata = {'model': name, 'numpy_version': str(np.__version__), 'threads': 1}
    if name == 'xgboost':
        print('[xgboost] importing XGBoost...', flush=True)
        import xgboost as xgb
        print('[xgboost] loading saved model...', flush=True)
        model = xgb.Booster(params={'nthread': 1, 'device': 'cpu'})
        model.load_model(folder / 'model.ubj')
        model.set_param({'nthread': 1, 'device': 'cpu'})
        training = read_json(folder / 'training_run.json')
        if model.feature_names != schema['feature_names'] or model.num_boosted_rounds() != training['saved_boosting_rounds']:
            raise ValueError('Unexpected XGBoost feature layout or tree count')
        metadata.update(version=str(xgb.__version__), saved_rounds=model.num_boosted_rounds())
        batch_size = 50_000
    else:
        print('[graphsage] importing PyTorch...', flush=True)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print('[graphsage] importing graph helpers...', flush=True)
        import torch_geometric
        from graphsage_trial import GraphBatchBuilder, TransactionGraphSAGE
        from prepare_graph import TemporalGraphIndex
        print('[graphsage] loading saved best-epoch model...', flush=True)
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        config = read_json(folder / 'training_config.json')
        training = read_json(folder / 'training_run.json')
        if saved['feature_count'] != schema['feature_count'] or saved['best_epoch'] != training['best_epoch']:
            raise ValueError('Unexpected GraphSAGE model checkpoint')
        model = TransactionGraphSAGE(saved['feature_count'], saved['branch_width'], saved['head_width'])
        model.load_state_dict(saved['model_state_dict'])
        model.eval()
        metadata.update(version=str(torch.__version__), pyg_version=str(torch_geometric.__version__),
                        best_epoch=saved['best_epoch'])
        batch_size = int(config['batch_size'])
        if batch_size < 1 or int(config['fanout']) < 1:
            raise ValueError('Invalid saved batch size or fanout')
    other = 'torch' if name == 'xgboost' else 'xgboost'
    if other in sys.modules:
        raise RuntimeError(f'Isolation failed: {other} was imported in the {name} worker')
    metadata['other_model_framework_loaded'] = False
    data = load_inputs(args.features, split, schema)
    n = len(data['X'])
    rows = np.unique(np.linspace(0, n - 1, min(512, n), dtype=np.int64)) if phase == 'check' else None
    count = len(rows) if rows is not None else n
    scores = np.empty(count, dtype=np.float32)
    graph = None
    try:
        if name == 'graphsage':
            graph = TemporalGraphIndex(args.graph)
            if graph.metadata['feature_schema_sha256'] != fingerprint(args.features / 'feature_schema.json'):
                raise ValueError('Graph index and features do not match')
            builder = GraphBatchBuilder(graph, data['X'], data['edges'], data['time_seconds'],
                                       read_json(folder / 'normalization.json'), fanout=int(config['fanout']))
        print(f'[{name}/{phase}] predicting {count:,} transactions...', flush=True)
        for start in range(0, count, batch_size):
            chosen = rows[start:start + batch_size] if rows is not None else np.arange(start, min(start + batch_size, n))
            if name == 'xgboost':
                scores[start:start + len(chosen)] = xgb_scores(model, data['X'][chosen], schema['feature_names'], 1)
            else:
                with torch.inference_mode():
                    scores[start:start + len(chosen)] = torch.sigmoid(model(builder.build(chosen))).numpy()
            if start == 0 or (start // batch_size + 1) % 500 == 0:
                print(f'[{name}/{phase}] {start + len(chosen):,}/{count:,}', flush=True)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('Invalid predictions')
        if phase == 'check':
            old_scores = np.load(folder / 'validation_scores.npy', mmap_mode='r', allow_pickle=False)
            old_ids = np.load(folder / 'validation_transaction_ids.npy', mmap_mode='r', allow_pickle=False)
            if old_scores.shape != (n,) or old_ids.shape != (n,):
                raise ValueError('Invalid saved validation arrays')
            np.testing.assert_array_equal(old_ids, data['transaction_ids'])
            # Single-thread inference may change floating-point reduction order slightly.
            np.testing.assert_allclose(scores, old_scores[rows], rtol=1e-5, atol=1e-7,
                                       err_msg=f'{name}: saved and reloaded predictions disagree')
            metadata.update(validation_end=int(data['time_seconds'][-1]), check='passed')
        else:
            np.save(args.work / f'{name}_test_scores.npy', scores, allow_pickle=False)
        write_json(args.work / f'{name}_{phase}.json', metadata)
    finally:
        if graph is not None:
            graph.close()
    print(f'[{name}/{phase}] complete', flush=True)


def launch_worker(args, name, phase, work, log):
    command = [sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
               '--worker', name, '--phase', phase, '--work', str(work),
               '--features', str(args.features.resolve()), '--graph', str(args.graph.resolve()),
               '--xgboost', str(args.xgboost.resolve()), '--graphsage', str(args.graphsage.resolve())]
    # A fresh interpreter executes each framework separately; this is not a forked worker pool.
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'\n--- {name}: {phase} ---\n')
        stream.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=os.environ.copy())
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code != 0:
        raise RuntimeError(f'{name}/{phase} failed (exit {code}). Diagnostic log: {log}')
    return read_json(work / f'{name}_{phase}.json')


def run(args):
    print('EVALUATOR STARTED — isolated model processes, one CPU thread each', flush=True)
    started = time.perf_counter()
    output, pending = args.output, args.output.with_name(args.output.name + '.building')
    if output.exists() or pending.exists():
        raise ValueError(f'{output} or {pending} exists. Nothing overwritten. If interrupted, send the error/output before retrying.')
    if args.threads != 1:
        raise ValueError('This crash-workaround version requires --threads 1')
    schema_path = args.features / 'feature_schema.json'
    schema, schema_hash = read_json(schema_path), fingerprint(schema_path)
    names = schema['feature_names']
    xrun = read_json(args.xgboost / 'training_run.json')
    gconfig = read_json(args.graphsage / 'training_config.json')
    grun = read_json(args.graphsage / 'training_run.json')
    xp = read_json(args.xgboost / 'alert_policy.json')
    gp = read_json(args.graphsage / 'alert_policy.json')
    xvalidation = read_json(args.xgboost / 'validation_metrics.json')
    gvalidation = read_json(args.graphsage / 'validation_metrics.json')
    normalization = read_json(args.graphsage / 'normalization.json')
    for folder, recorded in ((args.xgboost, xrun['feature_schema_sha256']),
                             (args.graphsage, gconfig['feature_schema_sha256'])):
        if recorded != schema_hash or fingerprint(folder / 'feature_schema.json') != schema_hash:
            raise ValueError(f'{folder}: feature build mismatch')
    if xrun['status'] != 'complete' or normalization['fit_split'] != 'train':
        raise ValueError('Incomplete model or invalid normalization')
    if len(names) != schema['feature_count'] or len(set(names)) != len(names):
        raise ValueError('Invalid feature names')
    if xp['selected_on'] != 'validation scores' or gp['selection_split'] != 'validation':
        raise ValueError('Alert rules must have been selected on validation')
    fraction = float(xp['validation_alert_fraction_target'])
    if not 0 < fraction < 1 or fraction != float(gp['rank_budget']['requested_alert_fraction']):
        raise ValueError('The two models must use the same validation alert budget')
    rules = {'xgboost': {'cutoff': float(xp['score_cutoff']), 'comparator': xp['comparator']},
             'graphsage': {key: gp['fixed_score_rule'][key] for key in ('cutoff', 'comparator')}}
    for name, report in (('xgboost', xvalidation), ('graphsage', gvalidation)):
        for key in ('cutoff', 'comparator'):
            if rules[name][key] != report['fixed_score_rule'][key]:
                raise ValueError(f'{name}: saved policy disagrees with validation report')
    graph_meta = read_json(args.graph / 'graph_manifest.json')
    if graph_meta['feature_schema_sha256'] != schema_hash:
        raise ValueError('Temporal graph and features do not match')
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / (output.name + '_diagnostics.log')
    with tempfile.TemporaryDirectory(prefix='fraud-evaluation-check-') as workdir:
        checks = {}
        for name in ('xgboost', 'graphsage'):
            checks[name] = launch_worker(args, name, 'check', Path(workdir), log)
    if checks['xgboost']['validation_end'] != checks['graphsage']['validation_end']:
        raise ValueError('Validation period mismatch')
    validation_end = checks['xgboost']['validation_end']
    print('Both model reload checks passed; no test labels loaded.', flush=True)
    if args.check_only:
        print(f'CHECK COMPLETE. Diagnostic log: {log}')
        return checks
    print('Freezing models and validation-selected rules.', flush=True)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'primary_model': 'xgboost', 'comparison_model': 'graphsage',
        'selection_basis': 'Earlier validation comparison; model roles fixed before test evaluation.',
        'validation_average_precision': {'xgboost': xvalidation['average_precision'],
                                         'graphsage': gvalidation['average_precision']},
        'feature_schema_sha256': schema_hash, 'source_csv_sha256': schema['source_csv_sha256'],
        'frozen_rules': rules, 'alert_fraction': fraction,
        'graphsage_best_epoch': checks['graphsage']['best_epoch'], 'graphsage_fanout': gconfig['fanout'],
        'xgboost_saved_rounds': checks['xgboost']['saved_rounds'],
        'worker_checks_and_versions': checks,
        'execution': 'sequential fresh processes; one CPU thread per worker',
        'models': {name: {'folder': str(folder.resolve()), 'files_sha256':
                   {file: fingerprint(folder / file) for file in files}}
                   for name, folder, files in (
                       ('xgboost', args.xgboost, ['model.ubj', 'alert_policy.json', 'training_run.json']),
                       ('graphsage', args.graphsage, ['model.pt', 'alert_policy.json', 'training_config.json', 'normalization.json']))},
        'notes': [
            'Main chronological test only; no test-driven model or threshold tuning.',
            'Earlier unlabeled test transactions enter history; current-time and future events are excluded.',
            'Top-k uses the complete test period offline; the frozen threshold is evaluated separately.',
            'The September 11-18 tail was separated after inspecting its unusual volume and label distribution. It is excluded here.',
            'Synthetic transaction labels; not verified mule-account detection or real-bank performance.',
            'Retrospective transaction scoring: both paid and received amounts are assumed observed.',
            'Both models already use historical endpoint features; this is a model comparison, not an isolated graph-feature ablation.',
            'One training seed per model; these results do not estimate variability across training seeds.',
        ],
    }
    pending.mkdir(parents=True)
    write_json(pending / 'evaluation_protocol.json', protocol)
    # Test inputs are opened only after the evaluation protocol is recorded.
    data = load_inputs(args.features, 'test', schema)
    if data['time_seconds'][0] <= validation_end:
        raise ValueError('Test overlaps validation')
    n = len(data['X'])
    print(f'Test transactions: {n:,}; offline top-budget alerts: {math.floor(n * fraction):,}', flush=True)
    for name in ('xgboost', 'graphsage'):
        launch_worker(args, name, 'test', pending, log)
    predictions = {name: np.load(pending / f'{name}_test_scores.npy', mmap_mode='r', allow_pickle=False)
                   for name in ('xgboost', 'graphsage')}
    np.save(pending / 'test_transaction_ids.npy', data['transaction_ids'], allow_pickle=False)
    print('Predictions saved. Loading test labels for evaluation only...', flush=True)
    labels = np.load(args.features / 'test_y.npy', mmap_mode='r', allow_pickle=False)
    if labels.shape != (n,) or np.count_nonzero(labels) != schema['laundering_by_split']['test']:
        raise ValueError('Test labels disagree with schema')
    reports = {name: test_metrics(labels, scores, data['transaction_ids'], fraction, rules[name])
               for name, scores in predictions.items()}
    write_json(pending / 'test_metrics.json', reports)
    table = []
    for name, report in reports.items():
        for policy in ('rank_budget', 'fixed_score_rule'):
            metrics = report[policy]
            table.append({'model': name, 'role': 'primary' if name == 'xgboost' else 'comparator',
                          'policy': policy, 'average_precision': report['average_precision'],
                          'pr_auc_trapezoidal': report['pr_auc_trapezoidal'],
                          **{k: metrics[k] for k in ('alerts', 'precision', 'recall', 'true_positives',
                             'false_positives', 'false_negatives', 'true_negatives',
                             'false_positives_per_1000_non_laundering')}})
    with (pending / 'test_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(pending / 'evaluation_run.json', {'status': 'complete',
               'seconds': time.perf_counter() - started, 'execution': 'isolated single-thread workers',
               'tail_evaluated': False, 'models_refitted': False})
    pending.rename(output)
    print('\nHELD-OUT TEST EVALUATION COMPLETE')
    for name, report in reports.items():
        print(f"\n{name.upper()}: AP={report['average_precision']:.6f}; PR-AUC={report['pr_auc_trapezoidal']:.6f}")
        for policy in ('rank_budget', 'fixed_score_rule'):
            m = report[policy]
            print(f"  {policy}: alerts={m['alerts']:,}; precision={m['precision']:.4%}; "
                  f"recall={m['recall']:.4%}; TP={m['true_positives']:,}; FP={m['false_positives']:,}")
    print(f'\nSaved to: {output}\nNo models were retrained. Tail arrays were not opened.')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features'))
    parser.add_argument('--graph', type=Path, default=Path('data/temporal_graph'))
    parser.add_argument('--xgboost', type=Path, default=Path('models/xgboost_baseline_v2'))
    parser.add_argument('--graphsage', type=Path, default=Path('models/graphsage_v2'))
    parser.add_argument('--output', type=Path, default=Path('reports/test_evaluation_v1'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--check-only', action='store_true', help='Check model reloads without opening test data')
    parser.add_argument('--worker', choices=['xgboost', 'graphsage'], help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=['check', 'test'], help=argparse.SUPPRESS)
    parser.add_argument('--work', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.phase is None or args.work is None:
            parser.error('worker requires phase and work')
        worker(args)
    else:
        run(args)
