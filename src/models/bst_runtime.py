from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.builders.bst_input_adapter import get_bone_pairs
from src.models.bst_model import BST, BST_0, BST_AP, BST_CG, BST_CG_AP


MERGED_SHUTTLESET_CLASSES = [
    "未知球種",
    "Top_放小球",
    "Top_擋小球",
    "Top_殺球",
    "Top_挑球",
    "Top_長球",
    "Top_平球",
    "Top_切球",
    "Top_推球",
    "Top_撲球",
    "Top_勾球",
    "Top_發短球",
    "Top_發長球",
    "Bottom_放小球",
    "Bottom_擋小球",
    "Bottom_殺球",
    "Bottom_挑球",
    "Bottom_長球",
    "Bottom_平球",
    "Bottom_切球",
    "Bottom_推球",
    "Bottom_撲球",
    "Bottom_勾球",
    "Bottom_發短球",
    "Bottom_發長球",
]

MERGED_SHUTTLESET_DISPLAY_CLASSES = [
    "未知球种",
    "放小球",
    "挡小球",
    "杀球",
    "挑球",
    "长球",
    "平球",
    "切球",
    "推球",
    "扑球",
    "勾球",
    "发短球",
    "发长球",
    "放小球",
    "挡小球",
    "杀球",
    "挑球",
    "长球",
    "平球",
    "切球",
    "推球",
    "扑球",
    "勾球",
    "发短球",
    "发长球",
]

_MODEL_CLASSES = {
    "BST_0": BST_0,
    "BST": BST,
    "BST_CG": BST_CG,
    "BST_AP": BST_AP,
    "BST_CG_AP": BST_CG_AP,
}


def _torch_load_state_dict(weight_path: str | Path) -> dict[str, Tensor]:
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"BST weight file not found: {path}")
    try:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(str(path), map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    elif isinstance(loaded, dict) and "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a state_dict-like object in {path}, got {type(loaded)!r}")
    return dict(loaded)


def infer_seq_len_from_state_dict(weight_path: str | Path) -> int | None:
    state_dict = _torch_load_state_dict(weight_path)
    candidates: list[tuple[str, int]] = []
    for key, value in state_dict.items():
        if not isinstance(value, Tensor) or value.ndim != 3:
            continue
        if "embedding_cross" in key or key.endswith("embedding_cross"):
            candidates.append((key, int(value.shape[1])))
        elif "embedding_tem" in key or key.endswith("embedding_tem"):
            if value.shape[1] > 1:
                candidates.append((key, int(value.shape[1] - 1)))
        elif "embedding_inter" in key or key.endswith("embedding_inter"):
            if value.shape[1] > 1:
                candidates.append((key, int(value.shape[1] - 1)))
    if not candidates:
        return None
    values = {seq_len for _, seq_len in candidates}
    if len(values) != 1:
        details = ", ".join(f"{key}=>{seq_len}" for key, seq_len in candidates)
        raise RuntimeError(f"Conflicting seq_len values in BST positional embeddings: {details}")
    return values.pop()


def _pose_in_dim(pose_style: str, *, n_joints: int = 17, in_channels: int = 2) -> int:
    n_bones = len(get_bone_pairs())
    match pose_style:
        case "J_only":
            extra = 0
        case "JnB_bone" | "JnB_interp":
            extra = 1
        case "Jn2B":
            extra = 2
        case _:
            raise ValueError(f"Unsupported pose_style: {pose_style!r}")
    return (n_joints + n_bones * extra) * in_channels


def _build_model_instance(model_name: str, seq_len: int, in_dim: int, n_classes: int) -> nn.Module:
    model_cls = _MODEL_CLASSES.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unsupported BST model_name: {model_name!r}. Expected one of {sorted(_MODEL_CLASSES)}")
    return model_cls(
        in_dim=in_dim,
        n_class=n_classes,
        seq_len=seq_len,
        depth_tem=2,
        depth_inter=1,
    )


def _state_dict_mismatch_report(model: nn.Module, state_dict: dict[str, Tensor]) -> str:
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    shape_mismatches = []
    for key in sorted(set(model_state) & set(state_dict)):
        value = state_dict[key]
        if hasattr(value, "shape") and tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatches.append((key, tuple(model_state[key].shape), tuple(value.shape)))

    lines: list[str] = []
    if missing:
        lines.append("missing keys:")
        lines.extend(f"  - {key}" for key in missing[:30])
        if len(missing) > 30:
            lines.append(f"  ... {len(missing) - 30} more")
    if unexpected:
        lines.append("unexpected keys:")
        lines.extend(f"  - {key}" for key in unexpected[:30])
        if len(unexpected) > 30:
            lines.append(f"  ... {len(unexpected) - 30} more")
    if shape_mismatches:
        lines.append("shape mismatches:")
        for key, model_shape, weight_shape in shape_mismatches[:30]:
            lines.append(f"  - {key}: model={model_shape}, weight={weight_shape}")
        if len(shape_mismatches) > 30:
            lines.append(f"  ... {len(shape_mismatches) - 30} more")
    return "\n".join(lines) if lines else "No key or shape mismatch detected before strict load."


def load_bst_weight_safely(model: nn.Module, weight_path: str | Path) -> None:
    state_dict = _torch_load_state_dict(weight_path)
    report = _state_dict_mismatch_report(model, state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"BST strict weight loading failed for {weight_path}.\n"
            f"{report}\n"
            f"Original PyTorch error:\n{exc}"
        ) from exc


def _candidate_seq_lens(weight_path: str | Path) -> list[int]:
    inferred = infer_seq_len_from_state_dict(weight_path)
    if inferred is not None:
        return [inferred]
    return [30, 100]


def _candidate_class_counts(weight_path: str | Path, n_classes: int) -> list[int]:
    counts = [int(n_classes)]
    filename = Path(weight_path).name.lower()
    fallback = 35 if int(n_classes) == 25 else 25
    if "merged" in filename and 25 not in counts:
        counts.insert(0, 25)
    if fallback not in counts:
        counts.append(fallback)
    return counts


def build_bst_model(
    weight_path: str | Path,
    model_name: str = "BST_CG_AP",
    pose_style: str = "JnB_bone",
    n_classes: int = 25,
) -> nn.Module:
    in_dim = _pose_in_dim(pose_style)
    attempts: list[str] = []
    for seq_len in _candidate_seq_lens(weight_path):
        for class_count in _candidate_class_counts(weight_path, n_classes):
            model = _build_model_instance(model_name, seq_len, in_dim, class_count)
            try:
                load_bst_weight_safely(model, weight_path)
            except RuntimeError as exc:
                attempts.append(
                    f"model_name={model_name}, pose_style={pose_style}, seq_len={seq_len}, "
                    f"n_classes={class_count}, in_dim={in_dim}\n{exc}"
                )
                continue
            model.bst_model_name = model_name  # type: ignore[attr-defined]
            model.bst_pose_style = pose_style  # type: ignore[attr-defined]
            model.bst_seq_len = seq_len  # type: ignore[attr-defined]
            model.bst_n_classes = class_count  # type: ignore[attr-defined]
            model.bst_in_dim = in_dim  # type: ignore[attr-defined]
            print(
                "BST model loaded: "
                f"model_name={model_name}, pose_style={pose_style}, seq_len={seq_len}, "
                f"n_classes={class_count}, in_dim={in_dim}"
            )
            if class_count != n_classes:
                print(
                    f"WARNING: loaded with n_classes={class_count}; requested {n_classes}. "
                    "The class label mapping may not match this weight."
                )
            return model

    raise RuntimeError(
        "Unable to build/load BST model. Weight structure, seq_len, n_classes, or model_name may be incompatible.\n\n"
        + "\n\n".join(attempts)
    )


def _as_tensor(data: np.ndarray | Tensor, *, device: str | torch.device, dtype: torch.dtype | None = None) -> Tensor:
    tensor = data if isinstance(data, Tensor) else torch.from_numpy(np.asarray(data))
    tensor = tensor.to(device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _validate_inference_shapes(
    human_pose: Tensor,
    shuttle: Tensor,
    pos: Tensor,
    video_len: Tensor,
    expected_seq_len: int | None,
) -> None:
    if human_pose.ndim != 4:
        raise ValueError(f"human_pose must have shape (B, T, 2, 72) after flatten, got {tuple(human_pose.shape)}")
    batch, steps, people, in_dim = human_pose.shape
    if people != 2 or in_dim != 72:
        raise ValueError(f"human_pose must have shape (B, T, 2, 72), got {tuple(human_pose.shape)}")
    if expected_seq_len is not None and steps != expected_seq_len:
        raise ValueError(f"human_pose T={steps} does not match model seq_len={expected_seq_len}")
    if tuple(shuttle.shape) != (batch, steps, 2):
        raise ValueError(f"shuttle must have shape {(batch, steps, 2)}, got {tuple(shuttle.shape)}")
    if tuple(pos.shape) != (batch, steps, 2, 2):
        raise ValueError(f"pos must have shape {(batch, steps, 2, 2)}, got {tuple(pos.shape)}")
    if tuple(video_len.shape) != (batch,):
        raise ValueError(f"video_len must have shape {(batch,)}, got {tuple(video_len.shape)}")


def run_bst_inference(
    model: nn.Module,
    human_pose: np.ndarray | Tensor,
    shuttle: np.ndarray | Tensor,
    pos: np.ndarray | Tensor,
    video_len: np.ndarray | Tensor,
    device: str | torch.device,
) -> dict[str, Any]:
    model = model.to(device)
    human_pose_t = _as_tensor(human_pose, device=device, dtype=torch.float32)
    shuttle_t = _as_tensor(shuttle, device=device, dtype=torch.float32)
    pos_t = _as_tensor(pos, device=device, dtype=torch.float32)
    video_len_t = _as_tensor(video_len, device=device).long()

    if human_pose_t.ndim == 5:
        if human_pose_t.shape[-2:] != (36, 2):
            raise ValueError(f"human_pose before flatten must end with (36, 2), got {tuple(human_pose_t.shape)}")
        human_pose_t = human_pose_t.view(*human_pose_t.shape[:-2], -1)

    _validate_inference_shapes(
        human_pose_t,
        shuttle_t,
        pos_t,
        video_len_t,
        getattr(model, "bst_seq_len", None),
    )
    model.eval()
    with torch.no_grad():
        logits = model(human_pose_t, shuttle_t, pos_t, video_len_t)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        top_prob, top_idx = torch.topk(prob, k=min(5, prob.shape[1]), dim=1)

    pred_ids = pred.detach().cpu().tolist()
    confidences = prob[torch.arange(prob.shape[0], device=prob.device), pred].detach().cpu().tolist()
    top5 = [
        [
            {
                "class_id": int(idx),
                "class_name": decode_merged_class(int(idx)) if prob.shape[1] == 25 else str(int(idx)),
                "probability": float(p),
            }
            for idx, p in zip(sample_idx.detach().cpu().tolist(), sample_prob.detach().cpu().tolist())
        ]
        for sample_idx, sample_prob in zip(top_idx, top_prob)
    ]
    pred_names = [
        decode_merged_class(int(idx)) if prob.shape[1] == 25 else str(int(idx))
        for idx in pred_ids
    ]
    result: dict[str, Any] = {
        "logits": logits.detach().cpu(),
        "prob": prob.detach().cpu(),
        "pred_id": pred_ids,
        "pred_name": pred_names,
        "confidence": confidences,
        "top5": top5,
    }
    if len(pred_ids) == 1:
        result.update(
            {
                "pred_id": pred_ids[0],
                "pred_name": pred_names[0],
                "confidence": confidences[0],
                "top5": top5[0],
            }
        )
    return result


def decode_merged_class(pred_id: int) -> str:
    if pred_id < 0 or pred_id >= len(MERGED_SHUTTLESET_CLASSES):
        raise ValueError(f"merged ShuttleSet class id must be in [0, 24], got {pred_id}")
    return MERGED_SHUTTLESET_CLASSES[pred_id]


def decode_merged_display_class(pred_id: int) -> str:
    if pred_id < 0 or pred_id >= len(MERGED_SHUTTLESET_DISPLAY_CLASSES):
        raise ValueError(f"merged ShuttleSet class id must be in [0, 24], got {pred_id}")
    return MERGED_SHUTTLESET_DISPLAY_CLASSES[pred_id]
