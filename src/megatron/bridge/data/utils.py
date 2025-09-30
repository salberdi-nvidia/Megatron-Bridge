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

from dataclasses import fields
from typing import Any, Callable, Dict, Optional, Type, Union

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.builders.hf_dataset import HFDatasetBuilder, HFDatasetConfig
from megatron.bridge.training.config import (
    DataloaderConfig,
    DatasetBuildContext,
    DatasetProvider,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    MockGPTDatasetConfig,
)
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.utils.common_utils import print_rank_0


def is_dataset_built_on_rank() -> bool:
    """Determines whether the dataset should be built on the current rank.

    Datasets are typically built only on the first and last pipeline stages
    and the first tensor parallel rank to save memory and avoid redundancy.

    Returns:
        True if the dataset should be built on the current rank, False otherwise.
    """
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def pretrain_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: BlendedMegatronDatasetConfig
) -> tuple[GPTDataset, GPTDataset, GPTDataset]:
    """Build pretraining train, validation, and test datasets.

    Uses BlendedMegatronDatasetBuilder to create GPTDataset or MockGPTDataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the blended Megatron dataset.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """

    if dataset_config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    # Build the dataset on all ranks for TP-replicated loading
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, lambda: True, dataset_config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def hf_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: HFDatasetConfig, tokenizer: MegatronTokenizer
) -> tuple[Any, Any, Any]:
    """Build train, validation, and test datasets from a Hugging Face dataset.

    Uses HFDatasetBuilder to create dataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the Hugging Face dataset.
        tokenizer: The MegatronTokenizer instance.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """
    print_rank_0(
        f"> building train, validation, and test datasets for Huggingface dataset {dataset_config.dataset_name} ..."
    )

    train_ds, valid_ds, test_ds = HFDatasetBuilder(
        tokenizer=tokenizer,
        **{
            field.name: getattr(dataset_config, field.name)
            for field in fields(dataset_config)
            if field not in fields(DataloaderConfig)
        },
    ).build()

    print_rank_0(f"> finished creating Huggingface dataset {dataset_config.dataset_name} ...")

    return train_ds, valid_ds, test_ds


def finetuning_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: FinetuningDatasetConfig, tokenizer: MegatronTokenizer
) -> tuple[Any, Any, Any]:
    """Build finetuning train, validation, and test datasets.

    Uses FinetuningDatasetBuilder to create dataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the finetuning dataset.
        tokenizer: The MegatronTokenizer instance.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """
    print_rank_0(
        f">building train, validation, and test datasets for Finetuning dataset from {dataset_config.dataset_root} ..."
    )

    train_ds, valid_ds, test_ds = FinetuningDatasetBuilder(
        tokenizer=tokenizer,
        **{
            field.name: getattr(dataset_config, field.name)
            for field in fields(dataset_config)
            if field not in fields(DataloaderConfig)
        },
    ).build()

    print_rank_0(f"> finished creating Finetuning dataset from {dataset_config.dataset_root} ...")

    return train_ds, valid_ds, test_ds


_REGISTRY: Dict[Type[Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig]], Callable] = {
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    MockGPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    HFDatasetConfig: hf_train_valid_test_datasets_provider,
    FinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
}


def get_dataset_provider(
    dataset_config: Union[FinetuningDatasetConfig, BlendedMegatronDatasetConfig, HFDatasetConfig, DatasetProvider],
) -> Callable:
    """Get the appropriate dataset provider function based on the config type.

    Supports both registry-based providers and protocol-based providers.

    Args:
        dataset_config: The dataset configuration object.

    Returns:
        The callable dataset provider function corresponding to the config type.
    """
    # Check if config implements the DatasetProvider protocol
    if isinstance(dataset_config, DatasetProvider):

        def protocol_adapter(
            train_val_test_num_samples: list[int],
            config: DatasetProvider,
            tokenizer: Optional[MegatronTokenizer] = None,
        ) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
            """Adapter function that bridges the protocol interface with the legacy interface."""
            context = DatasetBuildContext(
                train_samples=train_val_test_num_samples[0],
                valid_samples=train_val_test_num_samples[1],
                test_samples=train_val_test_num_samples[2],
                tokenizer=tokenizer,
            )
            return config.build_datasets(context)

        return protocol_adapter

    # Fall back to existing registry
    return _REGISTRY[type(dataset_config)]
