import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED
from argparse import ArgumentParser

import uproot
import awkward as ak
import numpy as np

import torch
from torch_geometric.data import Data

import random

SIM_VARS = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_vtxperp"]

T3_VARS = ["t3_pt", "t3_eta", "t3_phi"]
LS_INDEX = ["t3_lsIdx0", "t3_lsIdx1"]

LS_VARS = ["ls_dPhis", "ls_dPhiChanges", "ls_dAlphaInners", "ls_dAlphaOuters", "ls_dAlphaInnerOuters"]

MD_VARS = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "md_other_x", "md_other_y", "md_other_z", "md_dphi", "md_dphichange", "md_dz"]
MD_INDEX = ["ls_mdIdx0", "ls_mdIdx1"]

TARGET = ["t3_simIdx"]

FAKE_TARGET = ["t3_isFake"]

LAYER_INFO = ["md_layer"]
MD_SIMIDX = ["md_simIdx"]

ALL_COLUMNS = SIM_VARS + T3_VARS + LS_INDEX + LS_VARS + MD_VARS + MD_INDEX + TARGET + FAKE_TARGET + LAYER_INFO + MD_SIMIDX

MAX_T3_PT = 2000

class GraphBuilder:
    def __init__(self, input_paths, output_path, train_split=0.8):
        self.output_path = output_path
        self.train_split = train_split
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.train_path = os.path.join(self.output_path, "train")
        self.val_path = os.path.join(self.output_path, "val")
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

        if isinstance(input_paths, str):
            input_paths = [input_paths]
        self.input_trees = [uproot.open(p)["tree"] for p in input_paths]

    @staticmethod
    def _to_tensor(values, dtype=torch.float32):
        return torch.as_tensor(np.asarray(values), dtype=dtype)

    @staticmethod
    def _extract_graph_index(filename):
        if not (filename.startswith("graph_") and filename.endswith(".pt")):
            return None

        idx_str = filename[len("graph_"):-len(".pt")]
        if not idx_str.isdigit():
            return None
        return int(idx_str)

    def _load_existing_graph_indices(self):
        existing = set()

        for directory in (self.train_path, self.val_path):
            if not os.path.isdir(directory):
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    idx = self._extract_graph_index(entry.name)
                    if idx is not None:
                        existing.add(idx)

        return existing

    def _iter_selected_events(self, tree, num_events, selected_local_indices, chunk_size):
        """Yield selected events while reading ROOT arrays in chunks."""
        if not selected_local_indices:
            return

        selected = sorted(selected_local_indices)
        selected_ptr = 0
        n_selected = len(selected)

        for start in range(0, num_events, chunk_size):
            if selected_ptr >= n_selected:
                break

            stop = min(start + chunk_size, num_events)
            if selected[selected_ptr] >= stop:
                continue

            batch = tree.arrays(ALL_COLUMNS, entry_start=start, entry_stop=stop)
            while selected_ptr < n_selected and selected[selected_ptr] < stop:
                local_idx = selected[selected_ptr]
                offset = local_idx - start
                yield local_idx, batch[offset: offset + 1]
                selected_ptr += 1

    def process_event(self, event_data, idx, args):
        """Processes a single event and saves the graph."""
        try:
            if random.random() < self.train_split:
                output_file = os.path.join(self.train_path, f"graph_{idx}.pt")
            else:
                output_file = os.path.join(self.val_path, f"graph_{idx}.pt")

            if args.debug:
                ...
            elif not args.overwrite:
                train_exists = os.path.exists(os.path.join(self.train_path, f"graph_{idx}.pt"))
                val_exists = os.path.exists(os.path.join(self.val_path, f"graph_{idx}.pt"))
                if not (train_exists or val_exists):
                    pass
                else:
                    print(f"Graph {idx} already exists, skipping...")
                    return

            t3_df = ak.to_dataframe(event_data[T3_VARS])
            pt_mask = t3_df["t3_pt"] < MAX_T3_PT

            t3_features = t3_df[pt_mask].values

            ls_idx = ak.to_dataframe(event_data[LS_INDEX])[pt_mask].values
            md_idx = ak.to_dataframe(event_data[MD_INDEX]).values[ls_idx]

            ls_features = ak.to_dataframe(event_data[LS_VARS]).values[ls_idx]
            ls_features = ls_features.reshape(-1, 2 * len(LS_VARS))

            md_idx_flat = md_idx.reshape(-1, 4)
            md_features = ak.to_dataframe(event_data[MD_VARS]).values[md_idx_flat]

            md_features = md_features.reshape(-1, 4 * len(MD_VARS))

            md_layer = ak.to_dataframe(event_data[LAYER_INFO]).values[md_idx_flat]
            md_layer = md_layer.reshape(-1, 4)

            md_simIdx = ak.to_dataframe(event_data[MD_SIMIDX]).values[md_idx_flat]
            md_simIdx = md_simIdx.reshape(-1, 4)

            target_np = ak.to_dataframe(event_data[TARGET])[pt_mask].values

            if args.nofakes:
                fake_mask = ak.to_dataframe(event_data[FAKE_TARGET])[pt_mask].values.flatten() == 0
                node_features_np = np.concatenate([t3_features[fake_mask], ls_features[fake_mask], md_features[fake_mask]], axis=1)
                target_flat = self._to_tensor(target_np[fake_mask]).flatten()
                md_layer = self._to_tensor(md_layer[fake_mask])
                md_simIdx = self._to_tensor(md_simIdx[fake_mask])

            else:
                node_features_np = np.concatenate([t3_features, ls_features, md_features], axis=1)
                target_flat = self._to_tensor(target_np).flatten()
                target_flat[target_flat < 0] = -999999
                md_layer = self._to_tensor(md_layer)
                md_simIdx = self._to_tensor(md_simIdx)

            node_features = self._to_tensor(node_features_np)
            sim_features = self._to_tensor(ak.to_dataframe(event_data[SIM_VARS]).values)

            graph = Data(x=node_features, sim_index=target_flat, sim_features=sim_features, md_layer=md_layer, md_simIdx=md_simIdx)
            if args.debug:
                print(graph)
                return

            torch.save(graph, output_file)
            print(f"Processed graph {idx}")

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")

    def process_events_in_parallel(self, args):
        n_cpu = os.cpu_count() or 1
        n_workers = min(args.n_workers, max(1, n_cpu // 2)) if not args.debug else 1
        chunk_size = 1 if args.debug else max(1, args.chunk_size)
        max_pending = max(1, n_workers * 4)
        global_idx = 0
        skipped_existing = 0

        existing_indices = set()
        if not args.overwrite:
            existing_indices = self._load_existing_graph_indices()
            if existing_indices:
                print(f"Found {len(existing_indices)} existing graphs. Pre-skipping before ROOT I/O.")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            pending = set()
            for tree in self.input_trees:
                tree_entries = int(tree.num_entries)
                num_events = tree_entries if args.n_events == -1 else min(args.n_events, tree_entries)
                if args.debug:
                    num_events = 1

                tree_global_start = global_idx
                selected_local_indices = []
                for local_idx in range(num_events):
                    abs_idx = tree_global_start + local_idx
                    if args.overwrite or abs_idx not in existing_indices:
                        selected_local_indices.append(local_idx)
                    else:
                        skipped_existing += 1

                for local_idx, event_data in self._iter_selected_events(tree, num_events, selected_local_indices, chunk_size):
                    abs_idx = tree_global_start + local_idx
                    future = executor.submit(self.process_event, event_data, abs_idx, args)
                    pending.add(future)

                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for finished in done:
                            finished.result()

                    if args.debug:
                        break

                if args.debug:
                    break

                global_idx += num_events

            if pending:
                done, _ = wait(pending)
                for finished in done:
                    finished.result()

        if skipped_existing:
            print(f"Skipped {skipped_existing} existing graph indices before read/submit.")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input", type=str, nargs="+", required=True)
    argparser.add_argument("--output", type=str, default="./")
    argparser.add_argument("--n_workers", "-n", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing graphs")
    argparser.add_argument("--n_events", type=int, default=-1, help="Maximum number of events to process (-1 for all)")
    argparser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio")
    argparser.add_argument("--nofakes", action="store_true", help="Exclude fake hits from the graphs")
    argparser.add_argument("--chunk_size", type=int, default=256, help="Number of events to read per ROOT I/O chunk")
    argparser.add_argument("--seed", type=int, default=None, help="Optional random seed for train/val split")
    args = argparser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    training_data = GraphBuilder(args.input, args.output, args.split)
    training_data.process_events_in_parallel(args)