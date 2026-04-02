import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from Matcher.utils.logging_config import setup_logging
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = setup_logging(__name__)


def _repair_leaked_meta_init() -> bool:
    """Detect and fix a leaked accelerate ``init_empty_weights`` context.

    When transformers/accelerate loads a model with ``device_map``, it
    temporarily monkey-patches ``nn.Module.register_parameter`` so that
    every new parameter is placed on the ``meta`` device.  If the patch
    is not properly restored (e.g. due to nesting with BitsAndBytes
    quantization hooks), **all** subsequent ``nn.Module`` creations
    silently produce meta tensors.

    This function probes for the leak and, if found, replaces
    ``register_parameter`` (and ``register_buffer``) with clean
    re-implementations identical to the PyTorch originals.
    """
    probe = nn.Linear(1, 1, bias=False)
    leaked = probe.weight.is_meta
    del probe
    if not leaked:
        return False

    logger.warning(
        "Leaked accelerate init_empty_weights context detected – "
        "nn.Module.register_parameter is still patched to create meta "
        "tensors.  Restoring the original PyTorch implementation."
    )

    # ── restore register_parameter (PyTorch source-compatible) ──────
    def _clean_register_parameter(self, name: str, param) -> None:  # type: ignore[override]
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )
        if not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {torch.typename(name)}")
        if "." in name:
            raise KeyError('parameter name can\'t contain "."')
        if name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        if param is not None and not isinstance(param, torch.nn.Parameter):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter "
                f"'{name}' (torch.nn.Parameter or None required)"
            )
        modules = self.__dict__.get("_modules")
        if isinstance(modules, dict) and name in modules:
            raise KeyError(f"attribute '{name}' already exists as a submodule")
        self._parameters[name] = param

    nn.Module.register_parameter = _clean_register_parameter  # type: ignore[assignment]

    # ── restore register_buffer if it is also patched ───────────────
    buf_probe = nn.BatchNorm1d(1)
    if any(b is not None and b.is_meta for b in buf_probe._buffers.values()):
        def _clean_register_buffer(self, name: str, tensor, persistent: bool = True) -> None:  # type: ignore[override]
            if "_buffers" not in self.__dict__:
                raise AttributeError(
                    "cannot assign buffer before Module.__init__() call"
                )
            if not isinstance(name, str):
                raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
            if "." in name:
                raise KeyError('buffer name can\'t contain "."')
            if name == "":
                raise KeyError('buffer name can\'t be empty string ""')
            if name in self._parameters:
                raise KeyError(f"attribute '{name}' already exists as a parameter")
            if hasattr(self, name) and name not in self._buffers:
                raise KeyError(f"attribute '{name}' already exists")
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise TypeError(f"buffer must be a Tensor or None. Got {torch.typename(tensor)}")
            modules = self.__dict__.get("_modules")
            if isinstance(modules, dict) and name in modules:
                raise KeyError(f"attribute '{name}' already exists as a submodule")
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

        nn.Module.register_buffer = _clean_register_buffer  # type: ignore[assignment]
    del buf_probe

    # ── verify ──────────────────────────────────────────────────────
    verify = nn.Linear(1, 1, bias=False)
    if verify.weight.is_meta:
        del verify
        raise RuntimeError(
            "Failed to repair leaked accelerate init_empty_weights context. "
            "Try upgrading transformers/accelerate, or file a bug."
        )
    del verify
    logger.info("Successfully restored nn.Module.register_parameter.")
    return True


class LLMReranker:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: int = 0,
        torch_dtype=torch.float16,
        batch_size: int = 8,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        # Resolve device string
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            idx = int(device) if isinstance(device, int) else 0
            if idx < 0 or idx >= cuda_count:
                logger.warning(
                    f"LLMReranker: requested CUDA device {device} invalid; using 0 (num_gpus={cuda_count})."
                )
                idx = 0
            self.device_str = f"cuda:{idx}"
            # Ensure Accelerate/HF loaders use the selected GPU when device_map='auto'
            try:
                torch.cuda.set_device(idx)
            except Exception as e:
                logger.warning(f"Could not set CUDA device to {idx}: {e}")
        else:
            logger.warning("LLMReranker: CUDA not available; using CPU.")
            self.device_str = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.pad_token = self.tokenizer.unk_token or "<pad>"
            self.tokenizer.padding_side = "left"
        self._initialize_token_ids()
        self.model = self.load_model()
        self.model_lock = threading.Lock()

    def _initialize_token_ids(self):
        responses = ["Yes", "No"]
        token_ids = [
            self.tokenizer(response, add_special_tokens=False)["input_ids"]
            for response in responses
        ]
        self.applicable_token_id, self.not_applicable_token_id = [
            ids[0] for ids in token_ids
        ]

    def load_model(self):
        use_cuda = self.device_str.startswith("cuda")
        dtype = self.torch_dtype if use_cuda else torch.float32

        _repair_leaked_meta_init()

        logger.info("Loading reranker base model from %s", self.model_path)
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if use_cuda:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        model.tie_weights()
        self._raise_if_meta_tensors_remain(model, stage="after base model load")
        logger.info("Moving model to %s", self.device_str)
        model = model.to(self.device_str)

        ap = self.adapter_path
        if ap is not None and str(ap).strip() != "":
            adapter_dir = str(ap).strip()
            logger.info("Loading LoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(
                model, adapter_dir, torch_dtype=dtype
            )
            self._raise_if_meta_tensors_remain(model, stage="after adapter load")
        else:
            logger.info("No reranker LoRA adapter; using base model only.")

        model.eval()
        return model

    @staticmethod
    def _raise_if_meta_tensors_remain(model, stage: str) -> None:
        # Scan internal registries directly (same paths used by Module._apply/.to()).
        # named_parameters() can hide duplicates and miss tied/meta edge cases.
        remaining: List[str] = []
        for module_name, module in model.named_modules():
            for pname, param in module._parameters.items():  # pylint: disable=protected-access
                if param is not None and getattr(param, "is_meta", False):
                    full = f"{module_name}.{pname}" if module_name else pname
                    remaining.append(full)
            for bname, buf in module._buffers.items():  # pylint: disable=protected-access
                if buf is not None and getattr(buf, "is_meta", False):
                    full = f"{module_name}.{bname}" if module_name else bname
                    remaining.append(full)

        # Stable order and dedup for readable diagnostics.
        remaining = sorted(set(remaining))
        if remaining:
            preview = ", ".join(remaining[:20])
            raise RuntimeError(
                f"Model still has meta tensors {stage}: {preview}"
                + (" ..." if len(remaining) > 20 else "")
            )

    def preprocess_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def create_messages(self, patient_text: str, trial_text: str) -> List[Dict]:
        system_prompt = (
            "You are a clinical assistant tasked with determining whether the patient information (Statement A) "
            "provides enough details to evaluate whether the patient satisfies or violates the clinical "
            "trial eligibility criterion (Statement B). Respond with 'Yes' if Statement A contains sufficient "
            "information to make this evaluation, or 'No' if it does not."
        )
        return [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": " "},
            {
                "role": "user",
                "content": f"Statement A: {patient_text}\nStatement B: {trial_text}\n\n",
            },
        ]

    def process_batch(self, batch: List[tuple]) -> List[Dict]:
        batch_prompts = []
        for patient_text, trial_text in batch:
            messages = self.create_messages(
                self.preprocess_text(patient_text), self.preprocess_text(trial_text)
            )
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(prompt)
        inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        with self.model_lock:
            with torch.no_grad():
                outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probabilities = F.softmax(logits, dim=-1)
        applicable_probs = probabilities[:, self.applicable_token_id].tolist()
        return [
            {"llm_score": prob, "answer": "Yes" if prob > 0.5 else "No"}
            for prob in applicable_probs
        ]

    def rank_pairs(self, patient_trial_pairs: List[tuple]) -> List[Dict]:
        batches = [
            patient_trial_pairs[i : i + self.batch_size]
            for i in range(0, len(patient_trial_pairs), self.batch_size)
        ]
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in batches]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batches"
            ):
                results.extend(future.result())
        return results

