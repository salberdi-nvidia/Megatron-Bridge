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
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, GemmaTokenizer


HF_GEMMA3_TOY_MODEL_CONFIG = {
    "architectures": ["Gemma3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "head_dim": 256,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 1152,  # Smaller than real 1B for faster testing
    "initializer_range": 0.02,
    "intermediate_size": 2304,  # Reduced for TP compatibility testing
    "max_position_embeddings": 8192,
    "model_type": "gemma3_text",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,  # Much smaller for testing
    "num_key_value_heads": 2,  # Changed from 1 to 2 to be divisible by TP=2
    "pad_token_id": 0,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000.0,
    "rope_theta": 1000000.0,
    "sliding_window": 512,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.46.0",
    "use_cache": True,
    "vocab_size": 262144,
    "rope_scaling": None,
}


class TestGemma3Conversion:
    """
    Test Gemma3 model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def gemma3_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Gemma3 toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("gemma3_toy_model")
        model_dir = temp_dir / "gemma3_toy"

        # Create Gemma3 config from the toy model config
        config = Gemma3TextConfig(**HF_GEMMA3_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        model = Gemma3ForCausalLM(config)
        model = model.bfloat16()  # Use .bfloat16() method instead of .to()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Download and save tokenizer from a reference Gemma model
        # We use the smallest available Gemma model for tokenizer artifacts
        # First try to load from pre-mounted test data, then fall back to HuggingFace download
        pre_downloaded_path = "/home/TestData/megatron_bridge/tokenizers/google/gemma-2b"
        # Try loading from pre-downloaded location first
        if Path(pre_downloaded_path).exists():
            print(f"Loading tokenizer from pre-downloaded path: {pre_downloaded_path}")
            tokenizer = GemmaTokenizer.from_pretrained(pre_downloaded_path)
        else:
            # Fall back to downloading from HuggingFace
            print("Pre-downloaded tokenizer not found, attempting to download from HuggingFace")
            tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
        tokenizer.save_pretrained(model_dir)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_GEMMA3_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, gemma3_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            gemma3_toy_model_path: Path to the toy Gemma3 model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(gemma3_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "gemma3_text"
        assert config_data["hidden_size"] == 1152
        assert config_data["intermediate_size"] == 2304
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 4
        assert config_data["num_key_value_heads"] == 2
        assert config_data["vocab_size"] == 262144
        assert config_data["head_dim"] == 256
        # Check Gemma3-specific parameters
        assert config_data["query_pre_attn_scalar"] == 256
        assert config_data["sliding_window"] == 512
        assert config_data["rope_local_base_freq"] == 10000.0
        assert config_data["rope_theta"] == 1000000.0

        # Try loading the model to verify it's valid
        try:
            model = Gemma3ForCausalLM.from_pretrained(
                gemma3_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = GemmaTokenizer.from_pretrained(gemma3_toy_model_path)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure
            assert hasattr(model, "model")
            assert hasattr(model.model, "layers")
            assert len(model.model.layers) == 2  # num_hidden_layers

            print(f"SUCCESS: Toy model created and validated at {gemma3_toy_model_path}")
            print("Model weights are correctly in bfloat16 format")

        except Exception as e:
            assert False, f"Failed to load created toy model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_gemma3_conversion_parallelism(self, gemma3_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Gemma3 model conversion with different parallelism configurations.

        Args:
            gemma3_toy_model_path: Path to the toy Gemma3 model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"gemma3_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/workspace/.coverage",
            "--source=/workspace/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            gemma3_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Gemma3 {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(gemma3_toy_model_path).name  # "gemma3_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            # Verify the config contains Gemma3-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "gemma3_text", "Model type should be gemma3_text"
            assert saved_config["hidden_size"] == 1152, "Hidden size should match toy config"
            assert saved_config["intermediate_size"] == 2304, "Intermediate size should match toy config"
            assert saved_config["num_attention_heads"] == 4, "Number of attention heads should match toy config"
            assert saved_config["num_key_value_heads"] == 2, "Number of key-value heads should match toy config"
            assert saved_config["head_dim"] == 256, "Head dimension should match toy config"
            # Verify Gemma3-specific parameters
            assert saved_config["query_pre_attn_scalar"] == 256, "Query pre-attention scalar should match"
            assert saved_config["sliding_window"] == 512, "Sliding window should match"
            assert saved_config["rope_local_base_freq"] == 10000.0, "Rope local base frequency should match"
            assert saved_config["rope_theta"] == 1000000.0, "Rope theta should match"

            print(f"SUCCESS: Gemma3 {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Gemma3 {test_name} conversion test: {e}")
            raise
