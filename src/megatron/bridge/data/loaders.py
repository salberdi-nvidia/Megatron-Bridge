# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import torch
from megatron.core import mpu
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import RerunDataIterator
from torch.utils.data import DataLoader

from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler
from megatron.bridge.utils.common_utils import print_rank_0


def get_blend_and_blend_per_split(
    data_paths: Optional[list[str]] = None,
    data_args_path: Optional[str] = None,
    per_split_data_args_path: Optional[str] = None,
    train_data_paths: Optional[list[str]] = None,
    valid_data_paths: Optional[list[str]] = None,
    test_data_paths: Optional[list[str]] = None,
) -> tuple[Optional[list[str]], Optional[list[list[str]]]]:
    """Determine dataset blends from command-line arguments or config files.

    Parses different ways dataset paths/weights can be specified (single list,
    per-split lists, config files) and returns the blend information.

    Args:
        data_paths: List of paths/weights for a single blended dataset.
        data_args_path: Path to a file containing data paths/weights for a single blend.
        per_split_data_args_path: Path to a JSON file containing train/valid/test splits,
                                  each with its own list of paths/weights.
        train_data_paths: List of paths/weights specifically for the training split.
        valid_data_paths: List of paths/weights specifically for the validation split.
        test_data_paths: List of paths/weights specifically for the test split.

    Returns:
        A tuple (blend, blend_per_split):
        - blend: A list representing a single data blend, or None.
        - blend_per_split: A list containing blends for train, valid, test splits, or None.
                         Only one of `blend` or `blend_per_split` will be non-None.
    """
    use_data_path = data_paths is not None or data_args_path is not None
    use_per_split_data_path = (
        any(elt is not None for elt in [train_data_paths, valid_data_paths, test_data_paths])
        or per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if data_args_path is not None:
            assert data_paths is None
            with open(data_args_path, "r") as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert data_paths is not None
            blend = get_blend_from_list(data_paths)
    elif use_per_split_data_path:
        if per_split_data_args_path is not None:
            with open(per_split_data_args_path, "r") as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(train_data_paths),
                get_blend_from_list(valid_data_paths),
                get_blend_from_list(test_data_paths),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def cyclic_iter(iter: Iterable) -> Iterator:
    """Create an infinite iterator from a finite iterable."""
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples(cfg: ConfigContainer) -> tuple[int, int, int]:
    """Calculate the number of samples for train, validation, and test sets.

    Determines sample counts based on training mode (iteration-based vs sample-based),
    global batch size, and evaluation interval/iterations specified in the config.

    Args:
        cfg: The main configuration container.

    Returns:
        A tuple (train_samples, valid_samples, test_samples).
    """

    # Number of train samples - support both training modes
    if cfg.train.train_samples is not None:
        # Sample-based training - use direct sample count
        train_samples = cfg.train.train_samples
    else:
        # Iteration-based training - calculate from iterations
        train_samples = cfg.train.train_iters * cfg.train.global_batch_size

    # Validation and test samples calculation (unchanged logic)
    eval_iters = (cfg.train.train_iters // cfg.train.eval_interval + 1) * cfg.train.eval_iters
    test_iters = cfg.train.eval_iters

    return (
        train_samples,
        eval_iters * cfg.train.global_batch_size,
        test_iters * cfg.train.global_batch_size,
    )


def build_train_valid_test_datasets(
    cfg: ConfigContainer, build_train_valid_test_datasets_provider: Callable
) -> tuple[Any, Any, Any]:
    """Build train, validation, and test datasets using a provider function.

    Args:
        cfg: The main configuration container.
        build_train_valid_test_datasets_provider: A function that takes
            train_val_test_num_samples and dataset_config and returns the datasets.

    Returns:
        A tuple (train_dataset, valid_dataset, test_dataset).
    """
    train_valid_test_num_samples = get_train_valid_test_num_samples(cfg)
    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))

    # Show training mode information
    if cfg.train.train_samples is not None:
        print_rank_0(
            f" > Training mode: sample-based ({cfg.train.train_samples} samples -> {cfg.train.train_iters} iterations)"
        )
    else:
        print_rank_0(f" > Training mode: iteration-based ({cfg.train.train_iters} iterations)")

    return build_train_valid_test_datasets_provider(train_valid_test_num_samples, cfg.dataset)


def build_train_valid_test_data_loaders(
    cfg: ConfigContainer, train_state: TrainState, build_train_valid_test_datasets_provider: Callable
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build train, validation, and test data loaders.

    First builds the datasets using the provided provider function, then constructs
    PyTorch DataLoaders with appropriate sampling and configuration.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        build_train_valid_test_datasets_provider: A function to build the datasets.

    Returns:
        A tuple (train_dataloader, valid_dataloader, test_dataloader).
    """
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Backward compatibility for very old checkpoints that didn't track sample counts
    if train_state.step > 0 and train_state.consumed_train_samples == 0:
        # Only support iteration-based training for backward compatibility
        assert cfg.train.train_samples is None, "Backward compatibility only supported for iteration-based training"
        train_state.consumed_train_samples = train_state.step * cfg.train.global_batch_size
        print_rank_0(f"Backward compatibility: Setting consumed_train_samples to {train_state.consumed_train_samples}")

    if train_state.step > 0 and train_state.consumed_valid_samples == 0:
        if cfg.train.train_samples is None:  # iteration-based only
            train_state.consumed_valid_samples = (
                (train_state.step // cfg.train.eval_interval) * cfg.train.eval_iters * cfg.train.global_batch_size
            )
            print_rank_0(
                f"Backward compatibility: Setting consumed_valid_samples to {train_state.consumed_valid_samples}"
            )

    # Construct the data pipeline
    # Build datasets.
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        cfg=cfg, build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider
    )

    exit_signal = cfg.train.exit_signal

    def worker_init_fn(_):
        DistributedSignalHandler(exit_signal).__enter__()

    maybe_worker_init_fn = worker_init_fn if cfg.train.exit_signal_handler_for_dataloader else None

    # Build dataloders.
    train_dataloader = build_pretraining_data_loader(
        train_ds,
        train_state.consumed_train_samples,
        cfg.dataset.dataloader_type,
        cfg.train.micro_batch_size,
        cfg.dataset.num_workers,
        cfg.dataset.data_sharding,
        worker_init_fn=maybe_worker_init_fn,
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
    )
    if cfg.train.skip_train and cfg.train.eval_iters > 0:
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            0,
            cfg.dataset.dataloader_type,
            cfg.train.micro_batch_size,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
            collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
    elif cfg.train.eval_iters > 0:
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            train_state.consumed_valid_samples,
            "cyclic",
            cfg.train.micro_batch_size,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
            collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )

    if cfg.train.eval_iters > 0:
        test_dataloader = build_pretraining_data_loader(
            test_ds,
            0,
            cfg.dataset.dataloader_type,
            cfg.train.micro_batch_size,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
            collate_fn=test_ds.collate_fn if hasattr(test_ds, "collate_fn") else None,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and cfg.train.train_iters > 0
    do_valid = valid_dataloader is not None and cfg.train.eval_iters > 0
    do_test = test_dataloader is not None and cfg.train.eval_iters > 0
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    train_state.do_train = flags[0].item()
    train_state.do_valid = flags[1].item()
    train_state.do_test = flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
    cfg: ConfigContainer, train_state: TrainState, build_train_valid_test_datasets_provider: Callable
) -> tuple[Optional[RerunDataIterator], Optional[RerunDataIterator], Optional[RerunDataIterator]]:
    """Build train, validation, and test data iterators.

    Builds the data loaders first, then wraps them in appropriate iterators
    (e.g., RerunDataIterator, cyclic_iter) based on the configuration.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        build_train_valid_test_datasets_provider: A function to build the datasets.

    Returns:
        A tuple (train_data_iterator, valid_data_iterator, test_data_iterator).
    """

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        cfg=cfg,
        train_state=train_state,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
    )

    # Build iterators.
    dl_type = cfg.dataset.dataloader_type
    assert dl_type in ["single", "cyclic", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            else:
                return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator("cyclic", valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def setup_data_iterators(
    cfg: ConfigContainer,
    train_state: TrainState,
    model_length: int,
    train_valid_test_datasets_provider: Callable,
) -> tuple[
    Union[Optional[RerunDataIterator], list[Optional[RerunDataIterator]]],
    Union[Optional[RerunDataIterator], list[Optional[RerunDataIterator]]],
    Union[Optional[RerunDataIterator], list[Optional[RerunDataIterator]]],
]:
    """Set up data iterators, handling virtual pipeline parallelism if enabled.

    Calls `build_train_valid_test_data_iterators` potentially multiple times
    if virtual pipeline parallelism is used, creating separate iterators for each
    virtual stage.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        model_length: The number of model chunks (used for virtual pipeline parallelism).
        train_valid_test_datasets_provider: A function to build the datasets.

    Returns:
        A tuple (train_data_iterator, valid_data_iterator, test_data_iterator).
        Each element can be a single iterator or a list of iterators if virtual
        pipeline parallelism is enabled.
    """
    if cfg.model.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(model_length):
            iterators = build_train_valid_test_data_iterators(
                cfg=cfg,
                train_state=train_state,
                build_train_valid_test_datasets_provider=train_valid_test_datasets_provider,
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
            cfg=cfg,
            train_state=train_state,
            build_train_valid_test_datasets_provider=train_valid_test_datasets_provider,
        )

    return train_data_iterator, valid_data_iterator, test_data_iterator
