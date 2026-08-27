"""
Activation Hook Manager for Transformer Architectures.
Provides zero-VRAM leak residual stream interception with CPU offloading.
"""

from typing import Dict, List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn


class ActivationHookManager:
    """
    Manages forward hooks on transformer residual streams.
    Supports LLaMA, Mistral, Qwen, GPT-2, GPT-NeoX, and generic PyTorch modules.
    Ensures activations are detached and moved to CPU immediately to avoid VRAM leaks.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[int]] = None,
        token_position: int = -1,
        offload_to_cpu: bool = True,
    ):
        self.model = model
        self.total_layers = self._discover_layer_count()
        self.target_layers = target_layers if target_layers is not None else list(range(self.total_layers))
        self.token_position = token_position
        self.offload_to_cpu = offload_to_cpu
        
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, torch.Tensor] = {}
        self.full_sequence_activations: Dict[int, torch.Tensor] = {}
        self._is_registered = False

    def _discover_layer_count(self) -> int:
        """Dynamically identifies the number of transformer layers across architectures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return len(self.model.gpt_neox.layers)
        elif hasattr(self.model, "layers"):
            return len(self.model.layers)
        elif hasattr(self.model, "h"):
            return len(self.model.h)
        return 0

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Resolves layer module by index for various architecture implementations."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers[layer_idx]
        elif hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        elif hasattr(self.model, "h"):
            return self.model.h[layer_idx]
        else:
            raise AttributeError(f"Unsupported transformer architecture layout: {type(self.model)}")

    def _hook_factory(self, layer_idx: int) -> Callable:
        """Creates an isolated hook function for a specific layer index."""
        def hook(module: nn.Module, input_tensor: Tuple[torch.Tensor, ...], output_tensor: Union[Tuple[torch.Tensor, ...], torch.Tensor]):
            tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            
            # Tensor shape: (batch_size, sequence_length, hidden_dim)
            if not isinstance(tensor, torch.Tensor):
                return
            
            detached = tensor.detach()
            if self.offload_to_cpu:
                detached = detached.cpu()
                
            # Store full sequence and specific token position
            self.full_sequence_activations[layer_idx] = detached
            
            if detached.dim() >= 3:
                pos = self.token_position if self.token_position < detached.size(1) else -1
                self.activations[layer_idx] = detached[:, pos, :]
            else:
                self.activations[layer_idx] = detached

        return hook

    def register(self):
        """Registers hooks on all configured target layers."""
        if self._is_registered:
            return
        
        for layer_idx in self.target_layers:
            if layer_idx < self.total_layers:
                layer_module = self._get_layer_module(layer_idx)
                handle = layer_module.register_forward_hook(self._hook_factory(layer_idx))
                self.hooks.append(handle)
        
        self._is_registered = True

    def clear(self):
        """Clears captured activations from internal memory."""
        self.activations.clear()
        self.full_sequence_activations.clear()

    def remove(self):
        """Removes all registered PyTorch hooks and clears activation caches."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.clear()
        self._is_registered = False

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
