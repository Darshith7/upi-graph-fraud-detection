"""CPU temporal GraphSAGE training. Run from the project root.

Keep graphsage_trial.py, prepare_graph.py, and train_baseline.py in src/.
Uses train/validation only. Resume repeats an interrupted epoch.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch_geometric
from sklearn.metrics import average_precision_score

from graphsage_trial import (GraphBatchBuilder, TransactionGraphSAGE,
                            fit_normalizer, peak_memory_mib, write_json)
from prepare_graph import TemporalGraphIndex
from train_baseline import evaluate_validation


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_arrays(folder, split, schema):
    arrays = {key: np.load(folder / f'{split}_{key}.npy', mmap_mode='r',
                           allow_pickle=False)
              for key in ('X', 'y', 'edges', 'time_seconds', 'transaction_ids')}
    n = schema['rows_by_split'][split]
    shapes = {'X': (n, schema['feature_count']), 'y': (n,), 'edges': (n, 2),
              'time_seconds': (n,), 'transaction_ids': (n,)}
    for key, value in arrays.items():
        if value.shape != shapes[key]:
            raise ValueError(f'{split}/{key}: incorrect shape')
    y = arrays['y']
    positives = int(np.count_nonzero(y))
    if (not np.isin(y, [0, 1]).all() or not 0 < positives < n or
            positives != schema['laundering_by_split'][split]):
        raise ValueError(f'{split}: invalid labels or class counts')
    if np.any(np.diff(arrays['time_seconds']) < 0):
        raise ValueError(f'{split}: timestamps must be chronological')
    return arrays


def run(args):
    output = args.output
    pending = output.with_name(output.name + '.building')
    if output.exists():
        raise ValueError(f'{output} already exists; choose a new --output')
    checkpoint = pending / 'last_checkpoint.pt'
    if args.resume:
        if not checkpoint.is_file():
            raise ValueError(f'No resumable checkpoint at {checkpoint}')
        config = json.loads((pending / 'training_config.json').read_text())
        print('Resuming with saved settings; other training options are ignored.', flush=True)
    else:
        if pending.exists():
            raise ValueError(f'{pending} exists; use --resume if it has a checkpoint')
        config = {key: getattr(args, key) for key in
                  ('epochs', 'patience', 'batch_size', 'fanout', 'threads', 'seed',
                   'learning_rate', 'alert_fraction')}
        config.update(features=str(args.features.resolve()), graph=str(args.graph.resolve()),
                      torch_version=str(torch.__version__), pyg_version=str(torch_geometric.__version__),
                      selection='validation average precision', loss='unweighted binary cross entropy',
                      architecture='one-hop directional GraphSAGE, branch width 16, head width 64',
                      sampling='all training rows once per epoch in shuffled order',
                      history_policy='strictly earlier timestamps, including earlier unlabeled validation events')
    if any(config[k] < 1 for k in ('epochs', 'patience', 'batch_size', 'fanout', 'threads')):
        raise ValueError('Epochs, patience, batch size, fanout and threads must be positive')
    if not 0 < config['alert_fraction'] < 1 or not config['learning_rate'] > 0:
        raise ValueError('Invalid alert fraction or learning rate')
    if config['torch_version'] != str(torch.__version__) or config['pyg_version'] != str(torch_geometric.__version__):
        raise ValueError('Resume requires the original Torch/PyG versions')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    folder = Path(config['features'])
    schema_bytes = (folder / 'feature_schema.json').read_bytes()
    schema = json.loads(schema_bytes)
    schema_hash = hashlib.sha256(schema_bytes).hexdigest()
    if args.resume and config['feature_schema_sha256'] != schema_hash:
        raise ValueError('Feature schema changed since checkpoint')
    config['feature_schema_sha256'] = schema_hash
    train = load_arrays(folder, 'train', schema)
    valid = load_arrays(folder, 'validation', schema)
    if train['time_seconds'][-1] >= valid['time_seconds'][0]:
        raise ValueError('Training and validation periods overlap')
    if math.floor(len(valid['y']) * config['alert_fraction']) < 1:
        raise ValueError('Validation too small for alert fraction')
    with TemporalGraphIndex(Path(config['graph'])) as graph:
        if graph.metadata['feature_schema_sha256'] != schema_hash:
            raise ValueError('Graph and features come from different builds')
        if args.resume:
            normalizer = json.loads((pending / 'normalization.json').read_text())
        else:
            normalizer = fit_normalizer(train['X'], schema['feature_names'])
        model = TransactionGraphSAGE(schema['feature_count'], prior=float(train['y'].mean()))
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        loss_function = torch.nn.BCEWithLogitsLoss()
        builders = {name: GraphBatchBuilder(graph, data['X'], data['edges'],
                    data['time_seconds'], normalizer, fanout=config['fanout'])
                    for name, data in (('train', train), ('validation', valid))}
        if args.resume:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['rng'])
        else:
            pending.mkdir(parents=True)
            write_json(pending / 'training_config.json', config)
            write_json(pending / 'normalization.json', normalizer)
            (pending / 'feature_schema.json').write_bytes(schema_bytes)
            state = dict(completed_epoch=0, best_ap=-1.0, best_epoch=0, stale=0,
                         history=[], best_model=None, best_scores=None,
                         model=model.state_dict(), optimizer=optimizer.state_dict(),
                         rng=torch.get_rng_state())
            save_checkpoint(checkpoint, state)
        print(f"Training: {len(train['y']):,}; validation: {len(valid['y']):,}; "
              f"maximum epochs: {config['epochs']}; CPU threads: {config['threads']}", flush=True)
        batch_size = config['batch_size']
        for epoch in range(state['completed_epoch'] + 1, config['epochs'] + 1):
            if state['stale'] >= config['patience']:
                break
            started = time.perf_counter()
            order = np.random.default_rng(config['seed'] + epoch).permutation(len(train['y']))
            model.train()
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                batch = builders['train'].build(rows)
                labels = torch.from_numpy(np.array(train['y'][rows], dtype=np.float32))
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch), labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
                if start == 0 or (start // batch_size + 1) % 500 == 0:
                    print(f'Epoch {epoch}: trained {start + len(rows):,}/{len(order):,}', flush=True)
            train_seconds = time.perf_counter() - started
            model.eval()
            scores = np.empty(len(valid['y']), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(scores), batch_size):
                    rows = np.arange(start, min(start + batch_size, len(scores)))
                    scores[rows] = torch.sigmoid(model(builders['validation'].build(rows))).numpy()
                    if start == 0 or (start // batch_size + 1) % 500 == 0:
                        print(f'Epoch {epoch}: validated {start + len(rows):,}/{len(scores):,}', flush=True)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite validation predictions')
            ap = float(average_precision_score(valid['y'], scores))
            if ap > state['best_ap']:
                state.update(best_ap=ap, best_epoch=epoch, stale=0,
                             best_model=copy.deepcopy(model.state_dict()),
                             best_scores=torch.from_numpy(scores.copy()))
            else:
                state['stale'] += 1
            state['history'].append(dict(epoch=epoch, training_loss=total_loss / len(order),
                                         validation_ap=ap, train_seconds=train_seconds,
                                         epoch_seconds=time.perf_counter() - started))
            state.update(completed_epoch=epoch, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), rng=torch.get_rng_state())
            save_checkpoint(checkpoint, state)
            write_json(pending / 'training_history.json', state['history'])
            print(f'Epoch {epoch} saved. Validation AP: {ap:.6f}; '
                  f"best: {state['best_ap']:.6f} at epoch {state['best_epoch']}", flush=True)
        scores = state['best_scores'].numpy()
        metrics = evaluate_validation(valid['y'], scores, valid['transaction_ids'], config['alert_fraction'])
        metrics['notes'] = [note for note in metrics['notes']
                            if 'trees' not in note and 'XGBoost' not in note]
        metrics['notes'].insert(0, 'Validation selects the epoch using average precision; these are development results, not final test results.')
        save_checkpoint(pending / 'model.pt', {'model_state_dict': state['best_model'],
                        'feature_count': schema['feature_count'], 'branch_width': 16,
                        'head_width': 64, 'best_epoch': state['best_epoch']})
        np.save(pending / 'validation_scores.npy', scores, allow_pickle=False)
        np.save(pending / 'validation_transaction_ids.npy', valid['transaction_ids'], allow_pickle=False)
        write_json(pending / 'validation_metrics.json', metrics)
        write_json(pending / 'alert_policy.json', {'rank_budget': metrics['rank_budget'],
                   'fixed_score_rule': metrics['fixed_score_rule'], 'selection_split': 'validation'})
        write_json(pending / 'training_run.json', {'best_epoch': state['best_epoch'],
                   'completed_epochs': state['completed_epoch'], 'peak_process_memory_mib': peak_memory_mib(),
                   'test_and_tail_evaluated': False, 'config': config})
    pending.rename(output)
    rank = metrics['rank_budget']
    print('\nGRAPHSAGE TRAINING COMPLETE — VALIDATION RESULTS ONLY')
    print(f"Best epoch: {state['best_epoch']} / {state['completed_epoch']}")
    print(f"Average precision: {metrics['average_precision']:.6f}")
    print(f"Trapezoidal PR-AUC: {metrics['pr_auc_trapezoidal']:.6f}")
    print('Alert-budget results:', json.dumps(rank, indent=2))
    print(f'Saved to: {output}\nTest and tail labels/features were not opened.')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features'))
    parser.add_argument('--graph', type=Path, default=Path('data/temporal_graph'))
    parser.add_argument('--output', type=Path, default=Path('models/graphsage_v1'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--fanout', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--alert-fraction', type=float, default=0.005)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nInterrupted. If a checkpoint exists, add --resume to the same command.')
        print('Resume repeats the unfinished epoch; completed epochs are preserved.')
        raise SystemExit(130)


if __name__ == '__main__':
    main()
