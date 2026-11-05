"""Evaluate GraphSAGE and XGBoost on AMLSim Synthetic UPI & 3 Challenge Scenarios.

Scenarios evaluated:
1. Baseline
2. Challenge 1: Disguised Suspicious
3. Challenge 2: Benign Lookalike Hubs
4. Challenge 3: Partial History

Methodological Guarantees:
- Online past-only temporal features (zero future leakage: score first, add to graph second).
- 3-way temporal split on Baseline:
    Train:      Steps 1..75 (model fitting)
    Validation: Steps 76..90 (hyperparameter tuning, blend weight search, threshold calibration)
    Test:       Steps 91..149 (unseen out-of-time evaluation across all scenarios)
- Multi-epoch GraphSAGE training with early stopping on validation PR-AUC.
- Dynamic evolving graph simulation during test evaluation (matching deployed service).
- Comprehensive shared node mapping across all scenarios (distinct lookalike hub indices).
- Strict adherence to saved suppression mask in Challenge 3.
"""

import argparse
from array import array
from collections import defaultdict, deque
import faulthandler
import json
import math
import os
from pathlib import Path
import random
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
faulthandler.enable(all_threads=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import xgboost as xgb


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


W6, W24, W72 = 6 * 3600, 24 * 3600, 72 * 3600   # seconds (AMLSim: one step ~ 2h)

FEATURE_NAMES_CUMULATIVE = [
    "log1p_amount",
    "sender_log1p_out_count",
    "sender_log1p_in_count",
    "receiver_log1p_out_count",
    "receiver_log1p_in_count",
    "sender_log1p_seconds_since_out",
    "sender_log1p_seconds_since_in",
    "receiver_log1p_seconds_since_out",
    "receiver_log1p_seconds_since_in",
    "sender_log1p_unique_receivers",
    "receiver_log1p_unique_senders",
    "receiver_flow_through_flag",
    "insufficient_history_flag"
]

# Windowed features are stationary: they only look at the last 6h / 24h / 72h, so they do not
# keep growing as the stream gets longer (the cumulative counts did, which made scores drift).
FEATURE_NAMES_WINDOWED = [
    "log1p_amount",
    "sender_log1p_out_cnt_6h",
    "sender_log1p_out_cnt_24h",
    "sender_log1p_out_cnt_72h",
    "sender_log1p_in_cnt_24h",
    "receiver_log1p_in_cnt_6h",
    "receiver_log1p_in_cnt_24h",
    "receiver_log1p_in_cnt_72h",
    "receiver_log1p_out_cnt_24h",
    "sender_log1p_unique_receivers_24h",
    "receiver_log1p_unique_senders_24h",
    "receiver_log1p_unique_senders_72h",
    "sender_log1p_out_amount_24h",
    "receiver_log1p_in_amount_24h",
    "receiver_log1p_in_amount_72h",
    "receiver_flow_through_24h",
    "receiver_out_in_ratio_24h",
    "pair_log1p_repeat_cnt_72h",
    "pair_first_seen_flag",
    "sender_log1p_seconds_since_out",
    "receiver_log1p_seconds_since_in",
    "receiver_log1p_seconds_since_out",
    "insufficient_history_flag"
]

# Windowed features WITHOUT the pair-repeat features. AMLSim has almost no benign recurring payments,
# so a model that sees "same sender->receiver again within 72h" learns it as a laundering signal and
# then flags legitimate recurring payments (rent, subscriptions, splitting a bill).
PAIR_FEATURES = ("pair_log1p_repeat_cnt_72h", "pair_first_seen_flag")
FEATURE_NAMES_WINDOWED_NOPAIR = [n for n in FEATURE_NAMES_WINDOWED if n not in PAIR_FEATURES]

FEATURE_SETS = {"cumulative": FEATURE_NAMES_CUMULATIVE, "windowed": FEATURE_NAMES_WINDOWED,
                "windowed_nopair": FEATURE_NAMES_WINDOWED_NOPAIR}
DEFAULT_FEATURE_SET = "windowed"
FEATURE_NAMES = FEATURE_SETS[DEFAULT_FEATURE_SET]


def _scan_events(events, ts, target=None):
    """Aggregate a node's recent events strictly before-or-at `ts` (past-only).

    `events` holds (timestamp, counterparty, amount) roughly in time order. Returns
    (cnt6, cnt24, cnt72, amt24, amt72, uniq24, uniq72, target_cnt72)."""
    c6 = c24 = c72 = t72 = 0
    a24 = a72 = 0.0
    u24, u72 = set(), set()
    for t, other, amt in reversed(events):
        if t > ts:                # not yet happened relative to this transaction
            continue
        age = ts - t
        if age >= W72:
            break
        c72 += 1
        a72 += amt
        u72.add(other)
        if target is not None and other == target:
            t72 += 1
        if age < W24:
            c24 += 1
            a24 += amt
            u24.add(other)
            if age < W6:
                c6 += 1
    return c6, c24, c72, a24, a72, len(u24), len(u72), t72


class TemporalGraphFeatureTracker:
    """Computes past-only features for incoming transactions strictly before updating state."""

    def __init__(self, suppress_probability=0.0, feature_set: str = DEFAULT_FEATURE_SET):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
        self.feature_set = feature_set
        self.suppress_probability = suppress_probability
        self.out_counts = defaultdict(int)
        self.in_counts = defaultdict(int)
        self.last_out_time = {}
        self.last_in_time = {}
        self.out_neighbors = defaultdict(set)
        self.in_neighbors = defaultdict(set)
        self.out_events = defaultdict(deque)   # node -> (ts, receiver, amount), last 72h
        self.in_events = defaultdict(deque)    # node -> (ts, sender, amount), last 72h
        self.pair_counts = defaultdict(int)    # (sender, receiver) -> lifetime count
        self.rng = np.random.default_rng(42)

    @property
    def feature_names(self):
        return FEATURE_SETS[self.feature_set]

    def reset(self):
        self.out_counts.clear()
        self.in_counts.clear()
        self.last_out_time.clear()
        self.last_in_time.clear()
        self.out_neighbors.clear()
        self.in_neighbors.clear()
        self.out_events.clear()
        self.in_events.clear()
        self.pair_counts.clear()
        self.rng = np.random.default_rng(42)

    def extract_features(self, sender: int, receiver: int, amount: float, timestamp: int):
        s_out = self.out_counts[sender]
        s_in = self.in_counts[sender]
        r_out = self.out_counts[receiver]
        r_in = self.in_counts[receiver]

        s_t_out = math.log1p(max(0, timestamp - self.last_out_time[sender])) if sender in self.last_out_time else 0.0
        s_t_in = math.log1p(max(0, timestamp - self.last_in_time[sender])) if sender in self.last_in_time else 0.0
        r_t_out = math.log1p(max(0, timestamp - self.last_out_time[receiver])) if receiver in self.last_out_time else 0.0
        r_t_in = math.log1p(max(0, timestamp - self.last_in_time[receiver])) if receiver in self.last_in_time else 0.0

        insufficient_history = (s_out + s_in < 3) or (r_out + r_in < 3)
        log_amt = math.log1p(max(0.0, float(amount)))

        if self.feature_set == "cumulative":
            s_uniq_dst = len(self.out_neighbors[sender])
            r_uniq_src = len(self.in_neighbors[receiver])
            features = [
                log_amt, math.log1p(s_out), math.log1p(s_in), math.log1p(r_out), math.log1p(r_in),
                s_t_out, s_t_in, r_t_out, r_t_in,
                math.log1p(s_uniq_dst), math.log1p(r_uniq_src),
                float(r_in > 0 and r_out > 0), float(insufficient_history),
            ]
            return np.array(features, dtype=np.float32), insufficient_history

        so6, so24, so72, soa24, _, sou24, _, pair72 = _scan_events(self.out_events[sender], timestamp, target=receiver)
        _, si24, _, _, _, _, _, _ = _scan_events(self.in_events[sender], timestamp)
        ri6, ri24, ri72, ria24, ria72, riu24, riu72, _ = _scan_events(self.in_events[receiver], timestamp)
        _, ro24, _, _, _, _, _, _ = _scan_events(self.out_events[receiver], timestamp)

        features = [
            log_amt,
            math.log1p(so6), math.log1p(so24), math.log1p(so72),
            math.log1p(si24),
            math.log1p(ri6), math.log1p(ri24), math.log1p(ri72),
            math.log1p(ro24),
            math.log1p(sou24), math.log1p(riu24), math.log1p(riu72),
            math.log1p(soa24), math.log1p(ria24), math.log1p(ria72),
            float(ri24 > 0 and ro24 > 0),
            ro24 / (ri24 + 1.0),
            math.log1p(pair72),
            float(self.pair_counts[(sender, receiver)] == 0),
            s_t_out, r_t_in, r_t_out,
            float(insufficient_history),
        ]
        if self.feature_set == "windowed_nopair":
            keep = [i for i, n in enumerate(FEATURE_NAMES_WINDOWED) if n not in PAIR_FEATURES]
            features = [features[i] for i in keep]
        return np.array(features, dtype=np.float32), insufficient_history

    def update(self, sender: int, receiver: int, timestamp: int, force: bool = False, amount: float = 0.0):
        if not force and self.suppress_probability > 0.0 and self.rng.random() < self.suppress_probability:
            return
        self.out_counts[sender] += 1
        self.in_counts[receiver] += 1
        self.last_out_time[sender] = timestamp
        self.last_in_time[receiver] = timestamp
        self.out_neighbors[sender].add(receiver)
        self.in_neighbors[receiver].add(sender)
        self.pair_counts[(sender, receiver)] += 1
        if self.feature_set != "cumulative":
            amt = float(amount)
            out_q, in_q = self.out_events[sender], self.in_events[receiver]
            out_q.append((timestamp, receiver, amt))
            in_q.append((timestamp, sender, amt))
            horizon = timestamp - W72
            while out_q and out_q[0][0] < horizon:
                out_q.popleft()
            while in_q and in_q[0][0] < horizon:
                in_q.popleft()


class AMLSimGraphSAGE(nn.Module):
    def __init__(self, in_features=13, hidden_dim=32, out_dim=1):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, tx_feat, s_idx, d_idx):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        s_emb = h[s_idx]
        d_emb = h[d_idx]
        pair = torch.cat([tx_feat, s_emb, d_emb], dim=-1)
        return self.head(pair).squeeze(-1)


class LiveGraphState:
    """Causal graph state shared by training, evaluation and the live service.

    Node features are the RUNNING MEAN of the transaction feature vectors of every
    transaction the node has taken part in (bounded, unlike a raw sum that grows with
    degree). The edge list only ever contains transactions that were committed BEFORE the
    transaction being scored, so training, validation, test and serving all see the same
    kind of graph (past-only, no future edges, no self-edge).
    """

    def __init__(self, num_nodes: int, feat_dim: int):
        self.node_sum = torch.zeros((num_nodes, feat_dim), dtype=torch.float32)
        self.node_count = torch.zeros(num_nodes, dtype=torch.float32)
        self.src = torch.empty(0, dtype=torch.long)
        self.dst = torch.empty(0, dtype=torch.long)
        self._in_adj = None   # dst -> [src, ...] (with multiplicity); built lazily for serving

    @property
    def num_nodes(self) -> int:
        return self.node_sum.size(0)

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def node_x(self) -> torch.Tensor:
        return self.node_sum / self.node_count.clamp(min=1.0).unsqueeze(1)

    def edge_index(self) -> torch.Tensor:
        return torch.stack([self.src, self.dst])

    def ensure_nodes(self, num_nodes: int):
        """Grow the state so previously unseen accounts get their own (empty) node."""
        extra = num_nodes - self.num_nodes
        if extra > 0:
            self.node_sum = torch.cat([self.node_sum, torch.zeros((extra, self.node_sum.size(1)))])
            self.node_count = torch.cat([self.node_count, torch.zeros(extra)])

    def commit(self, s_idx: torch.Tensor, d_idx: torch.Tensor, feats: torch.Tensor):
        """Add scored transactions to the graph (call strictly AFTER scoring them)."""
        s_idx = s_idx.long().reshape(-1)
        d_idx = d_idx.long().reshape(-1)
        feats = feats.float().reshape(s_idx.numel(), -1)
        ones = torch.ones(s_idx.numel())
        self.node_sum.index_add_(0, s_idx, feats)
        self.node_sum.index_add_(0, d_idx, feats)
        self.node_count.index_add_(0, s_idx, ones)
        self.node_count.index_add_(0, d_idx, ones)
        self.src = torch.cat([self.src, s_idx])
        self.dst = torch.cat([self.dst, d_idx])
        if self._in_adj is not None:
            for s, d in zip(s_idx.tolist(), d_idx.tolist()):
                self._in_adj[d].append(s)

    def _adjacency(self):
        if self._in_adj is None:
            adj = defaultdict(list)
            for s, d in zip(self.src.tolist(), self.dst.tolist()):
                adj[d].append(s)
            self._in_adj = adj
        return self._in_adj

    def khop_inputs(self, seeds, hops: int = 2):
        """Exact inputs for a `hops`-layer GraphSAGE restricted to the seeds' receptive field.

        With mean aggregation, the layer-`hops` output of a seed depends only on its `hops`-hop
        IN-neighbourhood, so running the model on this subgraph gives the SAME logits as running
        it on the whole graph - at a cost that depends on local degree, not on graph size.
        Seeds >= num_nodes are brand-new accounts (zero features, no edges).
        Returns (node_x_local, edge_index_local, local_index_of_seeds)."""
        adj = self._adjacency()
        expanded, layer, nodes, edges = set(), set(seeds), set(seeds), []
        for _ in range(hops):
            nxt = set()
            for d in layer:
                if d in expanded:
                    continue
                expanded.add(d)
                for s in adj.get(d, ()):
                    edges.append((s, d))
                    nxt.add(s)
            nodes |= nxt
            layer = nxt
        order = sorted(nodes)
        local = {n: i for i, n in enumerate(order)}
        feat_dim = self.node_sum.size(1)
        x = torch.zeros((len(order), feat_dim), dtype=torch.float32)
        known = [n for n in order if n < self.num_nodes]
        if known:
            ids = torch.tensor(known, dtype=torch.long)
            vals = self.node_sum[ids] / self.node_count[ids].clamp(min=1.0).unsqueeze(1)
            x[torch.tensor([local[n] for n in known], dtype=torch.long)] = vals
        if edges:
            ei = torch.tensor([[local[s] for s, _ in edges], [local[d] for _, d in edges]], dtype=torch.long)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        return x, ei, [local[s] for s in seeds]

    def clone(self) -> "LiveGraphState":
        c = LiveGraphState(self.num_nodes, self.node_sum.size(1))
        c.node_sum = self.node_sum.clone()
        c.node_count = self.node_count.clone()
        c.src = self.src.clone()
        c.dst = self.dst.clone()
        return c

    def to_checkpoint(self) -> dict:
        return {
            "node_sum": self.node_sum, "node_count": self.node_count,
            "node_x": self.node_x(), "edge_index": self.edge_index(),
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "LiveGraphState":
        if "node_sum" not in ckpt or "node_count" not in ckpt:
            raise RuntimeError(
                "GraphSAGE checkpoint predates the causal running-mean graph state. "
                "Re-run src/evaluate_amlsim_challenges.py to regenerate models/amlsim_models."
            )
        st = cls(ckpt["node_sum"].size(0), ckpt["node_sum"].size(1))
        st.node_sum = ckpt["node_sum"].clone()
        st.node_count = ckpt["node_count"].clone()
        st.src = ckpt["edge_index"][0].clone()
        st.dst = ckpt["edge_index"][1].clone()
        return st


def group_by_step(steps: np.ndarray):
    """Index arrays of rows per time step, in chronological order."""
    return [np.where(steps == st)[0] for st in np.unique(steps)]


@torch.no_grad()
def score_gnn_causal(model: nn.Module, state: LiveGraphState, X: np.ndarray,
                     s_idx: np.ndarray, d_idx: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Score rows step by step: each step is scored against the graph of PREVIOUS steps only,
    then committed. Mutates `state`. Used identically for validation and test."""
    model.eval()
    scores = np.zeros(len(X), dtype=np.float32)
    X_t = torch.from_numpy(X).float()
    s_t = torch.as_tensor(s_idx, dtype=torch.long)
    d_t = torch.as_tensor(d_idx, dtype=torch.long)
    for idx in group_by_step(steps):
        ii = torch.as_tensor(idx, dtype=torch.long)
        out = model(state.node_x(), state.edge_index(), X_t[ii], s_t[ii], d_t[ii])
        scores[idx] = torch.sigmoid(out).numpy()
        state.commit(s_t[ii], d_t[ii], X_t[ii])
    return scores


def calibrate_alert_budget_threshold(val_scores: np.ndarray, alert_budget: float) -> float:
    """Threshold such that ~`alert_budget` of validation transactions are alerted.

    A fixed alert budget (an operational constraint: analyst capacity) is more stable than
    max-F1, which is very sensitive to prevalence / score drift between validation and test."""
    k = max(1, int(math.ceil(alert_budget * len(val_scores))))
    return float(np.sort(val_scores)[::-1][k - 1])


ALERT_WINDOW = 2000        # recent scores used to enforce the alert budget
ALERT_MIN_HISTORY = 100    # below this many recent scores, only the static cutoff applies


def effective_alert_threshold(recent_scores, static_threshold: float, alert_budget: float) -> float:
    """Alert cutoff = max(validated static cutoff, score at the top-`alert_budget` of RECENT scores).

    Features such as cumulative counts grow over time, so raw scores drift upward and a fixed
    cutoff ends up alerting on far more than the intended budget. The rolling term caps the
    alert rate near the budget under drift (it uses past scores only, so it is causal); the
    static term keeps quiet periods from alerting on the top slice of ordinary traffic."""
    n = len(recent_scores)
    if n < ALERT_MIN_HISTORY:
        return float(static_threshold)
    k = min(n - 1, max(0, int(math.floor((1.0 - alert_budget) * n))))
    rolling = float(np.partition(np.asarray(recent_scores, dtype=np.float64), k)[k])
    return max(float(static_threshold), rolling)


def rolling_budget_alerts(scores: np.ndarray, seed_scores, static_threshold: float,
                          alert_budget: float, window: int = ALERT_WINDOW) -> np.ndarray:
    """Replay the live alert policy over a chronological score stream."""
    buf = list(np.asarray(seed_scores, dtype=np.float64)[-window:])
    alerts = np.zeros(len(scores), dtype=bool)
    for i, sc in enumerate(scores):
        thr = effective_alert_threshold(buf, static_threshold, alert_budget)
        alerts[i] = sc >= thr
        buf.append(float(sc))
        if len(buf) > window:
            buf.pop(0)
    return alerts


def bootstrap_ap_difference(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                            n_boot: int = 200, seed: int = 0):
    """95% bootstrap CI of AP(score_a) - AP(score_b) (resampling transactions)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        if y[ii].sum() == 0:
            continue
        diffs.append(average_precision_score(y[ii], score_a[ii]) - average_precision_score(y[ii], score_b[ii]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": round(float(np.mean(diffs)), 4), "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4), "n_boot": len(diffs),
            "significant": bool(lo > 0 or hi < 0)}


def build_shared_node_mapping(scenario_files):
    all_node_ids = set()
    for p in scenario_files:
        df = pd.read_csv(Path(p))
        all_node_ids.update(df["sourceNodeId"].values)
        all_node_ids.update(df["targetNodeId"].values)

    sorted_nodes = sorted(list(all_node_ids))
    node2idx = {nid: i for i, nid in enumerate(sorted_nodes)}
    unknown_node_idx = len(node2idx)
    return node2idx, unknown_node_idx


def extract_scenario_features(csv_path: Path, suppress_prob: float = 0.0, feature_set: str = DEFAULT_FEATURE_SET):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    tracker = TemporalGraphFeatureTracker(suppress_probability=suppress_prob, feature_set=feature_set)
    has_saved_mask = "history_suppressed" in df.columns

    X = []
    insufficient_flags = []

    print(f"Extracting past-only features for {csv_path.name} ({len(df):,} rows)...")
    t0 = time.time()
    for row in df.itertuples(index=False):
        s_id = int(row.sourceNodeId)
        d_id = int(row.targetNodeId)
        amt = float(row.amount)
        t_sec = int(row.timestamp_seconds)

        feat, is_sparse = tracker.extract_features(s_id, d_id, amt, t_sec)
        X.append(feat)
        insufficient_flags.append(is_sparse)

        if has_saved_mask:
            is_suppressed = bool(getattr(row, "history_suppressed", 0))
            if not is_suppressed:
                tracker.update(s_id, d_id, t_sec, force=True, amount=amt)
        else:
            tracker.update(s_id, d_id, t_sec, amount=amt)

    X = np.stack(X)
    df["insufficient_history"] = insufficient_flags
    print(f"Extracted {len(X):,} feature vectors in {time.time() - t0:.2f}s")
    return df, X


def train_and_calibrate_models(
    base_df: pd.DataFrame,
    base_X: np.ndarray,
    node2idx: dict,
    unknown_node_idx: int,
    train_end_step: int = 75,
    val_end_step: int = 90,
    seed: int = 42,
    alert_budget: float = 0.05,
    gnn_epochs: int = 20,
    feature_names=None
):
    """Trains XGBoost & a CAUSAL GraphSAGE, tunes the blend on validation and sets the alert
    threshold from a fixed alert budget on validation.

    GraphSAGE is trained step by step: every training step is scored against the graph built
    from earlier steps only (running-mean node features, no future edges, no self edge) and
    validation is scored the same evolving way as test and live serving.
    """
    feature_names = list(feature_names or FEATURE_NAMES)
    seed_everything(seed)

    train_mask = (base_df["timestamp_step"] <= train_end_step).values
    val_mask = ((base_df["timestamp_step"] > train_end_step) & (base_df["timestamp_step"] <= val_end_step)).values

    train_df = base_df[train_mask].reset_index(drop=True)
    train_X = base_X[train_mask]
    train_y = train_df["is_laundering"].values

    val_df = base_df[val_mask].reset_index(drop=True)
    val_X = base_X[val_mask]
    val_y = val_df["is_laundering"].values

    print(f"Dataset split: Train (steps 1..{train_end_step}) = {len(train_df):,} rows ({train_y.sum():,} fraud), "
          f"Val (steps {train_end_step+1}..{val_end_step}) = {len(val_df):,} rows ({val_y.sum():,} fraud)")

    # 1. Train XGBoost
    print("Training XGBoost on train split...")
    dtrain = xgb.DMatrix(train_X, label=train_y, feature_names=feature_names, nthread=1)
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=feature_names, nthread=1)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 2.0,
        "tree_method": "hist",
        "nthread": 1,
        "seed": seed
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=120)
    xgb_val_scores = xgb_model.predict(dval)
    xgb_val_ap = float(average_precision_score(val_y, xgb_val_scores))
    print(f"XGBoost Validation AP: {xgb_val_ap:.4f}")

    # 2. Index arrays
    num_nodes = unknown_node_idx + 1
    s_tr = np.array([node2idx.get(s, unknown_node_idx) for s in train_df["sourceNodeId"].values], dtype=np.int64)
    d_tr = np.array([node2idx.get(d, unknown_node_idx) for d in train_df["targetNodeId"].values], dtype=np.int64)
    s_val = np.array([node2idx.get(s, unknown_node_idx) for s in val_df["sourceNodeId"].values], dtype=np.int64)
    d_val = np.array([node2idx.get(d, unknown_node_idx) for d in val_df["targetNodeId"].values], dtype=np.int64)
    train_steps = train_df["timestamp_step"].values
    val_steps = val_df["timestamp_step"].values
    train_groups = group_by_step(train_steps)

    tx_tr = torch.from_numpy(train_X).float()
    y_tr = torch.from_numpy(train_y).float()
    s_tr_t = torch.from_numpy(s_tr)
    d_tr_t = torch.from_numpy(d_tr)

    # 3. Train GraphSAGE causally: one optimisation step per time step, scored against the
    #    graph of earlier steps only.
    print(f"Training causal GraphSAGE ({gnn_epochs} epochs, one update per time step)...")
    gnn_model = AMLSimGraphSAGE(in_features=len(feature_names), hidden_dim=32, out_dim=1)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    best_gnn_ap = 0.0
    best_gnn_weights = None
    best_gnn_val_scores = None

    for epoch in range(1, gnn_epochs + 1):
        gnn_model.train()
        state = LiveGraphState(num_nodes, len(feature_names))
        for ii in train_groups:
            ii_t = torch.as_tensor(ii, dtype=torch.long)
            optimizer.zero_grad()
            out = gnn_model(state.node_x(), state.edge_index(), tx_tr[ii_t], s_tr_t[ii_t], d_tr_t[ii_t])
            loss = criterion(out, y_tr[ii_t])
            loss.backward()
            optimizer.step()
            state.commit(s_tr_t[ii_t], d_tr_t[ii_t], tx_tr[ii_t])  # commit AFTER the update/scoring

        # Validation: evolving graph, exactly like test / live serving
        val_probs = score_gnn_causal(gnn_model, state.clone(), val_X, s_val, d_val, val_steps)
        val_ap = float(average_precision_score(val_y, val_probs))
        val_auc = float(roc_auc_score(val_y, val_probs))
        print(f"  Epoch {epoch}/{gnn_epochs} -> Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}")
        if val_ap > best_gnn_ap:
            best_gnn_ap = val_ap
            best_gnn_weights = {k: v.clone() for k, v in gnn_model.state_dict().items()}
            best_gnn_val_scores = val_probs.copy()

    gnn_model.load_state_dict(best_gnn_weights)
    gnn_model.eval()
    print(f"GraphSAGE Best Validation AP: {best_gnn_ap:.4f}")

    # 4. Search optimal blend weight w_xgb in [0.0, 1.0] on validation set
    best_w = 0.70
    best_blend_val_ap = 0.0
    for w in np.linspace(0.0, 1.0, 21):
        blend_val = w * xgb_val_scores + (1.0 - w) * best_gnn_val_scores
        ap = float(average_precision_score(val_y, blend_val))
        if ap > best_blend_val_ap:
            best_blend_val_ap = ap
            best_w = round(float(w), 2)

    best_w_gnn = round(1.0 - best_w, 2)
    print(f"Validation-tuned Blend: {best_w:.2f} * XGBoost + {best_w_gnn:.2f} * GraphSAGE -> Val AP: {best_blend_val_ap:.4f}")

    # 5. Alert threshold from a FIXED ALERT BUDGET on validation (analyst-capacity constraint).
    #    Max-F1 is kept only as a reference: it transferred poorly from validation to test.
    calibrated_blend_val = best_w * xgb_val_scores + best_w_gnn * best_gnn_val_scores
    calibrated_threshold = round(calibrate_alert_budget_threshold(calibrated_blend_val, alert_budget), 4)
    val_alerts = calibrated_blend_val >= calibrated_threshold
    tp = int(np.sum(val_alerts & (val_y == 1)))
    val_precision = tp / max(1, int(val_alerts.sum()))
    val_recall = tp / max(1, int((val_y == 1).sum()))

    prec, rec, thresholds = precision_recall_curve(val_y, calibrated_blend_val)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)
    f1_idx = int(np.argmax(f1))
    ref_maxf1_threshold = round(float(thresholds[min(f1_idx, len(thresholds) - 1)]), 4)
    print(f"Alert-budget threshold ({alert_budget:.1%} of validation tx): {calibrated_threshold:.4f} "
          f"(Val Precision: {val_precision:.2%}, Val Recall: {val_recall:.2%}); "
          f"reference max-F1 threshold: {ref_maxf1_threshold:.4f}")

    # 6. Deployment graph state = everything up to the validation end (train + val committed).
    #    Node features do not depend on the model, so this is the same state the causal
    #    validation pass ended in.
    print("Pre-warming graph representation up to the validation end for deployment...")
    full_mask = (base_df["timestamp_step"] <= val_end_step).values
    full_df = base_df[full_mask].reset_index(drop=True)
    full_X = torch.from_numpy(base_X[full_mask]).float()
    s_full = torch.tensor([node2idx.get(s, unknown_node_idx) for s in full_df["sourceNodeId"].values], dtype=torch.long)
    d_full = torch.tensor([node2idx.get(d, unknown_node_idx) for d in full_df["targetNodeId"].values], dtype=torch.long)
    deploy_state = LiveGraphState(num_nodes, len(feature_names))
    for ii in group_by_step(full_df["timestamp_step"].values):
        ii_t = torch.as_tensor(ii, dtype=torch.long)
        deploy_state.commit(s_full[ii_t], d_full[ii_t], full_X[ii_t])

    calibration_info = {
        "best_w_xgb": best_w,
        "best_w_gnn": best_w_gnn,
        "calibrated_threshold": calibrated_threshold,
        "threshold_policy": "fixed_alert_budget_on_validation",
        "val_alert_budget": alert_budget,
        "reference_max_f1_threshold": ref_maxf1_threshold,
        "val_xgb_ap": round(xgb_val_ap, 4),
        "val_gnn_ap": round(best_gnn_ap, 4),
        "val_blend_ap": round(best_blend_val_ap, 4),
        "val_precision_at_thresh": round(float(val_precision), 4),
        "val_recall_at_thresh": round(float(val_recall), 4),
        "alert_window": ALERT_WINDOW,
        "val_score_reference": [round(float(v), 5) for v in calibrated_blend_val[-ALERT_WINDOW:]]
    }

    return xgb_model, gnn_model, deploy_state, calibration_info


def evaluate_scenario_evolving(
    name: str,
    df: pd.DataFrame,
    X: np.ndarray,
    xgb_model: xgb.Booster,
    gnn_model: nn.Module,
    init_state: LiveGraphState,
    node2idx: dict,
    unknown_node_idx: int,
    test_mask: np.ndarray,
    w_xgb: float = 0.70,
    w_gnn: float = 0.30,
    calibrated_threshold: float = 0.45,
    alert_fraction: float = 0.01,
    val_score_reference=None,
    alert_budget: float = 0.05,
    feature_names=None
):
    """Evaluates scenario on unseen test period using dynamic evolving graph (matching live service)."""
    feature_names = list(feature_names or FEATURE_NAMES)
    test_df = df[test_mask].reset_index(drop=True)
    test_X = X[test_mask]
    y_test = test_df["is_laundering"].values
    n_test = len(y_test)
    n_alerts = max(1, int(n_test * alert_fraction))

    # 1. XGBoost predictions on test features
    dtest = xgb.DMatrix(test_X, feature_names=feature_names, nthread=1)
    xgb_scores = xgb_model.predict(dtest)

    # 2. GraphSAGE evolving graph predictions: score step t against prior graph, then commit step t
    s_indices = np.array([node2idx.get(s, unknown_node_idx) for s in test_df["sourceNodeId"].values], dtype=np.int64)
    d_indices = np.array([node2idx.get(d, unknown_node_idx) for d in test_df["targetNodeId"].values], dtype=np.int64)

    # Score each step against the graph of PREVIOUS steps, then commit it (same code path as
    # validation and the live service).
    gnn_scores = score_gnn_causal(
        gnn_model, init_state.clone(), test_X, s_indices, d_indices, test_df["timestamp_step"].values
    )

    # 3. Deployed Blend scores
    blend_scores = w_xgb * xgb_scores + w_gnn * gnn_scores

    # Lookalike hub identification
    merchant_mask = test_df["receiver_upi"].str.contains("retail_hub", na=False).values
    payroll_mask = test_df["sender_upi"].str.contains("payroll", na=False).values
    lookalike_mask = merchant_mask | payroll_mask
    lookalike_count = int(np.sum(lookalike_mask))

    def compute_model_metrics(scores):
        ap = float(average_precision_score(y_test, scores))
        roc = float(roc_auc_score(y_test, scores))

        # Rank-budget (top 1%)
        top_rank_idx = np.argsort(scores)[::-1][:n_alerts]
        rank_mask = np.zeros(n_test, dtype=bool)
        rank_mask[top_rank_idx] = True

        tp_r = int(np.sum((y_test == 1) & rank_mask))
        fp_r = int(np.sum((y_test == 0) & rank_mask))
        fn_r = int(np.sum((y_test == 1) & (~rank_mask)))
        tn_r = int(np.sum((y_test == 0) & (~rank_mask)))
        precision_r = float(tp_r / (tp_r + fp_r)) if (tp_r + fp_r) > 0 else 0.0
        recall_r = float(tp_r / (tp_r + fn_r)) if (tp_r + fn_r) > 0 else 0.0
        hub_fps_r = int(np.sum(lookalike_mask[top_rank_idx]))

        # Calibrated Threshold Policy
        thresh_mask = scores >= calibrated_threshold
        tp_t = int(np.sum((y_test == 1) & thresh_mask))
        fp_t = int(np.sum((y_test == 0) & thresh_mask))
        fn_t = int(np.sum((y_test == 1) & (~thresh_mask)))
        tn_t = int(np.sum((y_test == 0) & (~thresh_mask)))
        precision_t = float(tp_t / (tp_t + fp_t)) if (tp_t + fp_t) > 0 else 0.0
        recall_t = float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 0.0
        alerts_t = int(np.sum(thresh_mask))
        alert_rate_t = float(alerts_t / n_test)
        hub_fps_t = int(np.sum(lookalike_mask & thresh_mask))

        return {
            "average_precision": round(ap, 4),
            "roc_auc": round(roc, 4),
            "rank_budget_1pct": {
                "alerts": n_alerts,
                "true_positives": tp_r,
                "false_positives": fp_r,
                "false_negatives": fn_r,
                "true_negatives": tn_r,
                "precision": round(precision_r, 4),
                "recall": round(recall_r, 4),
                "lookalike_hub_false_positives": hub_fps_r
            },
            "deployed_threshold_policy": {
                "threshold": calibrated_threshold,
                "alerts": alerts_t,
                "alert_rate": round(alert_rate_t, 4),
                "true_positives": tp_t,
                "false_positives": fp_t,
                "false_negatives": fn_t,
                "true_negatives": tn_t,
                "precision": round(precision_t, 4),
                "recall": round(recall_t, 4),
                "lookalike_hub_false_positives": hub_fps_t
            }
        }

    xgb_metrics = compute_model_metrics(xgb_scores)
    gnn_metrics = compute_model_metrics(gnn_scores)
    blend_metrics = compute_model_metrics(blend_scores)

    # Live alert policy (static cutoff AND rolling alert budget) replayed over the test stream
    order = np.lexsort((np.arange(n_test), test_df["timestamp_step"].values))
    policy_mask = np.zeros(n_test, dtype=bool)
    policy_mask[order] = rolling_budget_alerts(
        blend_scores[order], val_score_reference if val_score_reference is not None else [],
        calibrated_threshold, alert_budget
    )
    tp_p = int(np.sum((y_test == 1) & policy_mask))
    fp_p = int(np.sum((y_test == 0) & policy_mask))
    fn_p = int(np.sum((y_test == 1) & (~policy_mask)))
    # Re-weight negatives so the metrics are what they would be at a realistic laundering base rate
    # (the simulation has ~8% laundering; real AML prevalence is far lower).
    n_pos_t, n_neg_t = int(y_test.sum()), int(n_test - y_test.sum())
    prevalence_adjusted = {}
    for target in (0.01, 0.001):
        w_neg = (n_pos_t * (1.0 - target)) / (target * n_neg_t)
        sw = np.where(y_test == 1, 1.0, w_neg)
        tpr = tp_p / max(1, n_pos_t)
        fpr = fp_p / max(1, n_neg_t)
        prevalence_adjusted[str(target)] = {
            "average_precision": round(float(average_precision_score(y_test, blend_scores, sample_weight=sw)), 4),
            "policy_precision": round(float(tpr * target / max(1e-12, tpr * target + fpr * (1.0 - target))), 4),
            "policy_recall": round(float(tpr), 4),
            "false_alerts_per_true_alert": round(float(fpr * (1.0 - target) / max(1e-12, tpr * target)), 1),
        }
    blend_metrics["prevalence_adjusted"] = prevalence_adjusted
    blend_metrics["deployed_alert_policy"] = {
        "policy": "max(static validation cutoff, rolling top-budget cutoff)",
        "static_threshold": calibrated_threshold,
        "alert_budget": alert_budget,
        "alerts": int(policy_mask.sum()),
        "alert_rate": round(float(policy_mask.mean()), 4),
        "true_positives": tp_p,
        "false_positives": fp_p,
        "false_negatives": fn_p,
        "precision": round(tp_p / max(1, tp_p + fp_p), 4),
        "recall": round(tp_p / max(1, tp_p + fn_p), 4),
        "lookalike_hub_false_positives": int(np.sum(lookalike_mask & policy_mask))
    }

    # Error analysis
    top_gnn_idx = np.argsort(gnn_scores)[::-1][:n_alerts]
    fn_idx_gnn = np.where((y_test == 1) & (~np.isin(np.arange(n_test), top_gnn_idx)))[0]

    sample_missed_cases = []
    for idx in fn_idx_gnn[:5]:
        sample_missed_cases.append({
            "transaction_id": int(test_df.iloc[idx]["transaction_id"]),
            "sender_upi": test_df.iloc[idx]["sender_upi"],
            "receiver_upi": test_df.iloc[idx]["receiver_upi"],
            "amount": float(test_df.iloc[idx]["amount"]),
            "step": int(test_df.iloc[idx]["timestamp_step"]),
            "gnn_score": round(float(gnn_scores[idx]), 4),
            "xgb_score": round(float(xgb_scores[idx]), 4),
            "blend_score": round(float(blend_scores[idx]), 4),
            "reason": "Missed due to disguised timing/amount dispersion"
        })

    sample_fp_cases = []
    for idx in top_gnn_idx[y_test[top_gnn_idx] == 0][:5]:
        sample_fp_cases.append({
            "transaction_id": int(test_df.iloc[idx]["transaction_id"]),
            "sender_upi": test_df.iloc[idx]["sender_upi"],
            "receiver_upi": test_df.iloc[idx]["receiver_upi"],
            "amount": float(test_df.iloc[idx]["amount"]),
            "step": int(test_df.iloc[idx]["timestamp_step"]),
            "gnn_score": round(float(gnn_scores[idx]), 4),
            "xgb_score": round(float(xgb_scores[idx]), 4),
            "blend_score": round(float(blend_scores[idx]), 4),
            "is_lookalike_hub": bool(lookalike_mask[idx]),
            "reason": "Elevated fan-in triggering rank threshold"
        })

    return {
        "scenario": name,
        "test_events": n_test,
        "test_laundering": int(y_test.sum()),
        "test_laundering_rate": round(float(y_test.mean()), 4),
        "lookalike_hub_events": lookalike_count,
        "xgboost": xgb_metrics,
        "graphsage": gnn_metrics,
        "deployed_blend": blend_metrics,
        # Does the GNN actually help? 95% bootstrap CI of AP(blend) - AP(XGBoost alone).
        "blend_minus_xgboost_ap": bootstrap_ap_difference(y_test, blend_scores, xgb_scores),
        "sample_missed_fraud_cases": sample_missed_cases,
        "sample_false_alarm_cases": sample_fp_cases
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AMLSim challenges with evolving graph simulation")
    parser.add_argument("--challenges-dir", default="data/amlsim_challenges", help="Challenge directory")
    parser.add_argument("--output-dir", default="reports/amlsim_challenges", help="Report output directory")
    parser.add_argument("--train-split-step", type=int, default=75, help="Train split step")
    parser.add_argument("--val-split-step", type=int, default=90, help="Validation split step")
    parser.add_argument("--alert-fraction", type=float, default=0.01, help="Rank alert budget fraction")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--deployed-alert-budget", type=float, default=0.05,
                        help="Fraction of validation transactions the deployed threshold may alert on "
                             "(analyst capacity). The deployed cutoff is the validation score at this budget.")
    parser.add_argument("--gnn-epochs", type=int, default=20, help="Causal GraphSAGE training epochs")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default=DEFAULT_FEATURE_SET,
                        help="windowed (stationary, default) or cumulative (legacy, drifts over time)")
    parser.add_argument("--models-dir", default="models/amlsim_models", help="Where to save deployed models")
    args = parser.parse_args()

    seed_everything(args.seed)
    challenges_dir = Path(args.challenges_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_file = challenges_dir / "baseline.csv"
    c1_file = challenges_dir / "challenge_1_disguised.csv"
    c2_file = challenges_dir / "challenge_2_lookalike.csv"
    c3_file = challenges_dir / "challenge_3_partial_history.csv"

    print("Building comprehensive shared node mapping across all scenarios...")
    node2idx, unknown_node_idx = build_shared_node_mapping([base_file, c1_file, c2_file, c3_file])
    print(f"Mapped {len(node2idx):,} unique accounts (Reserved Unknown Index: {unknown_node_idx}).")

    # Step 1: Feature extraction for baseline
    feature_names = FEATURE_SETS[args.feature_set]
    base_df, base_X = extract_scenario_features(base_file, feature_set=args.feature_set)

    # Step 2: Model training, blend weight search, and threshold calibration
    xgb_model, gnn_model, deploy_state, calib = train_and_calibrate_models(
        base_df, base_X, node2idx, unknown_node_idx,
        train_end_step=args.train_split_step,
        val_end_step=args.val_split_step,
        seed=args.seed,
        alert_budget=args.deployed_alert_budget,
        gnn_epochs=args.gnn_epochs,
        feature_names=feature_names
    )

    val_score_reference = calib.pop("val_score_reference")   # kept in the checkpoint, not the JSON report

    # Save trained models for runtime service
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(models_dir / "xgboost_amlsim.ubj"))
    torch.save({
        "state_dict": gnn_model.state_dict(),
        **deploy_state.to_checkpoint(),   # node_sum, node_count, node_x (running mean), edge_index
        "node2idx": node2idx,
        "unknown_node_idx": unknown_node_idx,
        "feature_names": feature_names,
        "feature_set": args.feature_set,
        "calibration": {**calib, "val_score_reference": val_score_reference},
        "seed": args.seed
    }, str(models_dir / "graphsage_amlsim.pt"))
    print(f"Saved trained models to {models_dir}")

    # Step 3: Evaluate all 4 Scenarios on unseen test period (step > 90)
    scenarios = [
        ("Baseline (Standard AMLSim)", base_file, 0.0),
        ("Challenge 1 (Disguised Suspicious)", c1_file, 0.0),
        ("Challenge 2 (Benign Lookalike Hubs)", c2_file, 0.0),
        ("Challenge 3 (Partial History 50%)", c3_file, 0.50),
    ]

    all_results = {
        "metadata": {
            "evaluation_seed": args.seed,
            "feature_set": args.feature_set,
            "train_split_step": args.train_split_step,
            "val_split_step": args.val_split_step,
            "test_split_step": args.val_split_step,
            "rank_alert_fraction": args.alert_fraction,
            "deployed_score_blend": f"{calib['best_w_xgb']:.2f} * xgb + {calib['best_w_gnn']:.2f} * gnn",
            "deployed_threshold": calib["calibrated_threshold"],
            "calibration_summary": calib
        },
        "scenarios": {}
    }

    for name, path, supp_prob in scenarios:
        print(f"\n==========================================")
        print(f"Evaluating Scenario: {name}")
        print(f"==========================================")
        df, X = extract_scenario_features(path, suppress_prob=supp_prob, feature_set=args.feature_set)
        test_mask = (df["timestamp_step"] > args.val_split_step).values
        result = evaluate_scenario_evolving(
            name, df, X, xgb_model, gnn_model,
            deploy_state, node2idx, unknown_node_idx, test_mask,
            w_xgb=calib["best_w_xgb"],
            w_gnn=calib["best_w_gnn"],
            calibrated_threshold=calib["calibrated_threshold"],
            alert_fraction=args.alert_fraction,
            val_score_reference=val_score_reference,
            alert_budget=args.deployed_alert_budget,
            feature_names=feature_names
        )
        all_results["scenarios"][name] = result

        xgb_res = result["xgboost"]
        gnn_res = result["graphsage"]
        bld_res = result["deployed_blend"]
        print(f"XGBoost   -> AP: {xgb_res['average_precision']:.4f}, Recall@1%: {xgb_res['rank_budget_1pct']['recall']:.2%}, Hub FPs: {xgb_res['rank_budget_1pct']['lookalike_hub_false_positives']}")
        print(f"GraphSAGE -> AP: {gnn_res['average_precision']:.4f}, Recall@1%: {gnn_res['rank_budget_1pct']['recall']:.2%}, Hub FPs: {gnn_res['rank_budget_1pct']['lookalike_hub_false_positives']}")
        print(f"Blend     -> AP: {bld_res['average_precision']:.4f}, Recall@1%: {bld_res['rank_budget_1pct']['recall']:.2%}, Hub FPs: {bld_res['rank_budget_1pct']['lookalike_hub_false_positives']}")

    # Write JSON metrics
    metrics_file = output_dir / "challenge_metrics.json"
    metrics_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {metrics_file}")

    # Generate Markdown Summary
    md_lines = [
        "# AMLSim Synthetic UPI & 3 Controlled Challenge Cases Report",
        "",
        "## 1. Experimental Overview",
        "- **Base Dataset**: Synthetic UPI transaction graph generated via IBM AMLSim (20,000 nodes, 120,558 transactions, 200 cycles, 200 fan-in patterns).",
        f"- **Train Period**: Steps 1 to {args.train_split_step} (past-only online features, multi-epoch causal GraphSAGE training).",
        f"- **Validation Period**: Steps {args.train_split_step + 1} to {args.val_split_step} (blend weight grid search & threshold calibration).",
        f"- **Evaluation Period**: Steps {args.val_split_step + 1} to 149 (unseen out-of-time test stream).",
        f"- **Rank-budget metric**: top {args.alert_fraction * 100:.1f}% of evaluated transactions (Section 2).",
        f"- **Validation-Calibrated Blend**: `{calib['best_w_xgb']:.2f} * xgb_score + {calib['best_w_gnn']:.2f} * gnn_score` with production alert cutoff at `{calib['calibrated_threshold']:.4f}`.",
        "- **Causality**: XGBoost features are past-only; GraphSAGE is trained and validated on causal snapshots (each step sees only earlier steps, running-mean node features) and test/serving use the identical evolving-graph code path.",
        "",
        "## 2. Quantitative Comparison Table (Rank Budget @ 1%)",
        "",
        "| Scenario | Evaluated Tx | Fraud Tx | XGBoost AP | GraphSAGE AP | Blend AP | XGB Recall @ 1% | GraphSAGE Recall @ 1% | Blend Recall @ 1% | Lookalike Hub FPs (XGB / GNN / Blend) |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ]

    for name, r in all_results["scenarios"].items():
        xgb_r = r["xgboost"]
        gnn_r = r["graphsage"]
        bld_r = r["deployed_blend"]
        md_lines.append(
            f"| **{name}** | {r['test_events']:,} | {r['test_laundering']:,} | "
            f"{xgb_r['average_precision']:.4f} | {gnn_r['average_precision']:.4f} | {bld_r['average_precision']:.4f} | "
            f"{xgb_r['rank_budget_1pct']['recall']:.2%} | {gnn_r['rank_budget_1pct']['recall']:.2%} | {bld_r['rank_budget_1pct']['recall']:.2%} | "
            f"{xgb_r['rank_budget_1pct']['lookalike_hub_false_positives']} / {gnn_r['rank_budget_1pct']['lookalike_hub_false_positives']} / {bld_r['rank_budget_1pct']['lookalike_hub_false_positives']} |"
        )

    md_lines += [
        "",
        f"## 3. Deployed Alert Policy: score >= max({calib['calibrated_threshold']:.4f}, rolling top-{calib['val_alert_budget']:.0%} cutoff over the last {ALERT_WINDOW} scores)",
        "",
        "| Scenario | Evaluated Tx | Static Cutoff | Deployed Alerts | Alert Rate | True Positives | False Positives | Precision | Recall | Hub FPs |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ]

    for name, r in all_results["scenarios"].items():
        p = r["deployed_blend"]["deployed_alert_policy"]
        md_lines.append(
            f"| **{name}** | {r['test_events']:,} | >={p['static_threshold']:.4f} | "
            f"{p['alerts']:,} | {p['alert_rate']:.2%} | {p['true_positives']:,} | {p['false_positives']:,} | "
            f"{p['precision']:.2%} | {p['recall']:.2%} | {p['lookalike_hub_false_positives']} |"
        )

    sc = all_results["scenarios"]
    base_r = sc["Baseline (Standard AMLSim)"]
    c1_r = sc["Challenge 1 (Disguised Suspicious)"]
    c2_r = sc["Challenge 2 (Benign Lookalike Hubs)"]
    c3_r = sc["Challenge 3 (Partial History 50%)"]

    def ap(r, m):
        return r[m]["average_precision"]

    def pol(r):
        return r["deployed_blend"]["deployed_alert_policy"]

    def static_pol(r):
        return r["deployed_blend"]["deployed_threshold_policy"]

    md_lines += [
        "",
        "## 4. Does the GraphSAGE component help? (95% bootstrap CI of AP(blend) - AP(XGBoost alone))",
        "",
        "| Scenario | Delta AP | 95% CI | Significant |",
        "|---|---|---|---|"
    ]
    for name, r in sc.items():
        d = r["blend_minus_xgboost_ap"]
        md_lines.append(f"| **{name}** | {d['mean']:+.4f} | [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] | {'yes' if d['significant'] else 'no'} |")

    hub_events = c2_r["lookalike_hub_events"]
    hub_fp = pol(c2_r)["lookalike_hub_false_positives"]
    md_lines += [
        "",
        "## 5. Analysis of the Three Challenge Scenarios (numbers generated from this run)",
        "",
        "### Challenge 1: Disguised Suspicious Behavior",
        "- **Test**: Launderers split transfers into micro-ticket sizes (INR 115-485) and introduce randomized delays (+3 to +13 steps) strictly within split boundaries.",
        f"- **Result**: Blend AP {ap(base_r, 'deployed_blend'):.4f} (baseline) -> {ap(c1_r, 'deployed_blend'):.4f}; "
        f"recall at the deployed threshold {pol(base_r)['recall']:.1%} -> {pol(c1_r)['recall']:.1%}. "
        f"XGBoost AP {ap(base_r, 'xgboost'):.4f} -> {ap(c1_r, 'xgboost'):.4f}, GraphSAGE AP {ap(base_r, 'graphsage'):.4f} -> {ap(c1_r, 'graphsage'):.4f}.",
        "",
        "### Challenge 2: Benign Lookalike Hubs",
        "- **Test**: Legitimate high fan-in merchant QR hubs and high fan-out corporate payroll accounts were injected with collision-free transaction IDs.",
        f"- **Result**: {hub_fp} of {hub_events} benign hub events ({(hub_fp / hub_events if hub_events else 0):.1%}) are alerted at the deployed threshold. "
        "Pure-behaviour features cannot fully separate a busy legitimate merchant from a collection mule; a verified-merchant registry is needed for that (see the consumer copilot).",
        "",
        "### Challenge 3: Partial / Missing History",
        "- **Test**: 50% of earlier edges were withheld before scoring to simulate cold-start / unobserved cross-bank history.",
        f"- **Result**: Blend AP {ap(base_r, 'deployed_blend'):.4f} (baseline) -> {ap(c3_r, 'deployed_blend'):.4f}; "
        f"alert rate {pol(base_r)['alert_rate']:.2%} -> {pol(c3_r)['alert_rate']:.2%}, recall {pol(base_r)['recall']:.1%} -> {pol(c3_r)['recall']:.1%}. "
        "The `insufficient_history` flag is raised so the API can warn operators instead of giving unwarranted confidence.",
        "",
        "## 6. Threshold transfer (validation -> test)",
        f"- Policy: `{calib['threshold_policy']}` with a {calib['val_alert_budget']:.1%} validation alert budget -> cutoff {calib['calibrated_threshold']:.4f}.",
        f"- Validation: precision {calib['val_precision_at_thresh']:.1%}, recall {calib['val_recall_at_thresh']:.1%}, alert rate {calib['val_alert_budget']:.1%}.",
        f"- Test (baseline), static cutoff only: precision {static_pol(base_r)['precision']:.1%}, recall {static_pol(base_r)['recall']:.1%}, alert rate {static_pol(base_r)['alert_rate']:.2%} "
        "(scores drift upward over time because cumulative-count features grow, so a fixed cutoff over-alerts).",
        f"- Test (baseline), deployed policy with rolling budget: precision {pol(base_r)['precision']:.1%}, recall {pol(base_r)['recall']:.1%}, alert rate {pol(base_r)['alert_rate']:.2%}.",
        f"- Reference only: the previous max-F1 cutoff would have been {calib['reference_max_f1_threshold']:.4f}.",
        "",
        "## 7. Known limitations",
        "- All data here is simulated (AMLSim). The 'UPI' handles are labels attached to simulated accounts, not real UPI traffic.",
        "- Challenge scenarios are generated by this project's own scripts, so they test the model against the authors' own assumptions.",
        "- Laundering prevalence (~8%) is far above real-world AML base rates, so precision numbers are optimistic."
    ]

    summary_file = output_dir / "challenge_summary.md"
    summary_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved markdown summary to {summary_file}")


if __name__ == "__main__":
    main()
