#!/usr/bin/env python3

import argparse
import os
import json
import time

# Must be set before torch initializes CUDA. Masked events vary in size, and
# expandable segments greatly reduce allocator fragmentation for them.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import pytorch_lightning as pl

# Use Tensor Cores on A30 — free ~10-20% speedup on matmuls
torch.set_float32_matmul_precision('medium')

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from torch_geometric.loader import DataLoader

from dataset import PCDataset
from helpers import set_seed, load_config


class ParticleTrackingDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mask_kwargs = {
                "eta_mask_center": self.config["eta_mask_center"],
                "phi_mask_center": self.config["phi_mask_center"],
                "eta_mask_half_width": self.config["eta_mask_half_width"],
                "phi_mask_half_width": self.config["phi_mask_half_width"],
            }
            self.train_dataset = PCDataset(
                self.config['train_data_dir'],
                subset=self.config.get('train_subset', None),
                **mask_kwargs,
            )
            self.val_dataset = PCDataset(
                self.config['val_data_dir'],
                subset=self.config.get('val_subset', None),
                **mask_kwargs,
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False,
        )
    
    def get_feature_dims(self):
        """Get feature dimensions from the training dataset."""
        return self.train_dataset.get_feature_dims()


def describe_events(batch):
    """
    Describe the events in a batch as key=value pairs: file name, number of T3s after
    masking, and mask center. With several events, values are comma-separated lists.
    """
    files = ",".join(batch.filename)
    n_t3s = ",".join(str(n) for n in torch.diff(batch.ptr).tolist())
    eta_centers = ",".join(f"{c:.6g}" for c in batch.eta_mask_center.tolist())
    phi_centers = ",".join(f"{c:.6g}" for c in batch.phi_mask_center.tolist())
    centers = f"eta_center={eta_centers} phi_center={phi_centers}"
    return f"file={files} n_t3s={n_t3s} {centers}"


def format_metrics(metrics):
    """Format a dict of metrics as 'name=value name=value'."""
    return " ".join(f"{name}={float(value):.4g}" for name, value in metrics.items())


class PrintProgress(pl.Callback):
    """Print plain progress lines, and save them to train.log in the run's directory."""

    def __init__(self):
        self.log_file = None
        self.epoch_start = None

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            path = os.path.join(trainer.log_dir, "train.log")
            self.log_file = open(path, "a")

    def teardown(self, trainer, pl_module, stage):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def write(self, line):
        print(line, flush=True)
        if self.log_file is not None:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        # Wall time of the epoch, including validation in epochs where it runs
        epoch_s = time.perf_counter() - self.epoch_start
        timing = f"n_batches={trainer.num_training_batches} epoch_s={epoch_s:.1f}"
        self.write(f"stage=epoch epoch={trainer.current_epoch} {timing}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        epoch = trainer.current_epoch
        n_batches = trainer.num_training_batches
        events = describe_events(batch)
        metrics = format_metrics(trainer.progress_bar_metrics)
        step = f"epoch={epoch} batch={batch_idx + 1} n_batches={n_batches}"
        self.write(f"stage=train {step} {events} {metrics}")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val = {k: v for k, v in metrics.items() if k.startswith("val_")}
        self.write(f"stage=val epoch={epoch} {format_metrics(val)}")


class Trainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        data_module = ParticleTrackingDataModule(self.config)
        data_module.setup('fit')
        feature_dims = data_module.get_feature_dims()
        
        model_type = self.config.get('model', 'transformer')
        if model_type == 'transformer':
            from model import TransformerLightningModule
            model = TransformerLightningModule(
                self.config,
                input_dim=feature_dims['node_features_dim']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if self.config.get('compile', False):
            model.model = torch.compile(model.model, mode='reduce-overhead')
        
        tb_logger = TensorBoardLogger(
            save_dir=self.config['output_dir'],
            name='lightning_logs',
            default_hp_metric=True,
        )
        
        debug_mode = self.config.get('debug', False)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
            filename='ckpt-{epoch:03d}-{val_loss:.4f}',
            monitor='train_loss' if debug_mode else 'val_loss',
            mode='min',
            save_top_k=-1 if debug_mode else 3,
            save_last=True,
            every_n_epochs=1 if debug_mode else self.config.get('save_every', 5),
            save_on_train_epoch_end=True,
            verbose=True,
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 20),
            mode='min',
            verbose=True,
        )
        
        lr_monitor = LearningRateMonitor(
            logging_interval='step',
            log_momentum=True,
        )
        
        model_summary = ModelSummary(
            max_depth=2,
        )
        
        callbacks = [checkpoint_callback, lr_monitor, model_summary, PrintProgress()]
        if not debug_mode:
            callbacks.append(early_stopping)

        num_gpus = len(self.config.get('gpus', [1])) if isinstance(self.config.get('gpus'), list) else self.config.get('gpus', 1)
        accelerator = 'gpu' if torch.cuda.is_available() and num_gpus > 0 else 'cpu'

        # Optional timing report, written to fit-profile.txt in the run's directory
        if self.config['profiler'] is None:
            profiler = None
        elif self.config['profiler'] == 'simple':
            profiler = SimpleProfiler(filename='profile')
        else:
            raise ValueError(f"Unknown profiler: {self.config['profiler']}")

        trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            callbacks=callbacks,
            logger=[tb_logger],
            accelerator=accelerator,
            devices=(self.config.get('gpus', [1]) if isinstance(self.config.get('gpus'), list) else self.config.get('gpus', 1)),
            strategy="auto",
            enable_progress_bar=False,
            check_val_every_n_epoch=self.config.get('check_val_every_n_epoch', 5),
            log_every_n_steps=self.config.get('log_every_n_steps', 50),
            enable_checkpointing=True,
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            gradient_clip_algorithm=self.config.get('gradient_clip_algorithm', 'norm'),
            accumulate_grad_batches=self.config.get('accumulate_grad_batches', 1),
            precision=self.config.get('precision', 32),
            profiler=profiler,
        )

        ckpt_path = None
        if self.config.get('resume'):
            ckpt_path = self.config['resume']
            print(f"Resuming training from {ckpt_path}")

        trainer.fit(model, data_module, ckpt_path=ckpt_path)
        
        if trainer.is_global_zero:
            final_model_path = os.path.join(tb_logger.log_dir, 'final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            print(f"Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Graph Neural Network for Particle Tracking')
    parser.add_argument('--config', type=str,
                       help='Path to YAML config file')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError("A configuration file must be provided with --config")
    set_seed(config['seed'])
    os.makedirs(config['output_dir'], exist_ok=True)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()