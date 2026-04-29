from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from src.builders.bst_input_adapter import (
    build_jnb_bone,
    make_seq_len_same,
    normalize_joints,
    prepare_bst_batch,
)
from src.models.bst_runtime import (
    build_bst_model,
    decode_merged_display_class,
    infer_seq_len_from_state_dict,
    run_bst_inference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BST_WEIGHT = PROJECT_ROOT / "assets" / "weights" / "bst" / "bst_CG_AP_JnB_bone_merged_10.pt"


class BSTInputAdapterTest(unittest.TestCase):
    def test_display_class_names_are_simplified_and_prefix_free(self) -> None:
        self.assertEqual(decode_merged_display_class(0), "未知球种")
        self.assertEqual(decode_merged_display_class(1), "放小球")
        self.assertEqual(decode_merged_display_class(14), "挡小球")
        self.assertEqual(decode_merged_display_class(17), "长球")
        self.assertNotIn("_", decode_merged_display_class(24))

    def _fake_pose_inputs(self, frames: int = 45) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        bboxes = np.zeros((frames, 2, 4), dtype=np.float32)
        bboxes[:, 0] = np.array([120.0, 80.0, 360.0, 560.0], dtype=np.float32)
        bboxes[:, 1] = np.array([760.0, 180.0, 1020.0, 690.0], dtype=np.float32)
        bboxes += rng.normal(0.0, 3.0, size=bboxes.shape).astype(np.float32)
        bboxes[..., 2] = np.maximum(bboxes[..., 2], bboxes[..., 0] + 80.0)
        bboxes[..., 3] = np.maximum(bboxes[..., 3], bboxes[..., 1] + 160.0)

        joints = np.zeros((frames, 2, 17, 2), dtype=np.float32)
        for t in range(frames):
            for p in range(2):
                x1, y1, x2, y2 = bboxes[t, p]
                joints[t, p, :, 0] = rng.uniform(x1 + 10.0, x2 - 10.0, size=17)
                joints[t, p, :, 1] = rng.uniform(y1 + 10.0, y2 - 10.0, size=17)
        joints[3, 0, 9] = 0.0

        shuttle = rng.uniform([0.0, 0.0], [1280.0, 720.0], size=(frames, 2)).astype(np.float32)
        pos = rng.uniform(0.0, 1.0, size=(frames, 2, 2)).astype(np.float32)
        pos[:, 0, 1] = np.minimum(pos[:, 0, 1], pos[:, 1, 1])
        pos[:, 1, 1] = np.maximum(pos[:, 0, 1] + 0.01, pos[:, 1, 1])
        return joints, bboxes, shuttle, pos

    def test_shapes_through_jnb_batch_and_flatten(self) -> None:
        target_len = infer_seq_len_from_state_dict(BST_WEIGHT) if BST_WEIGHT.exists() else 30
        self.assertIsNotNone(target_len)
        joints_pixel, bboxes, shuttle, pos = self._fake_pose_inputs()
        joints_norm = normalize_joints(joints_pixel, bboxes, center_align=True)
        self.assertTrue(np.all(joints_norm[3, 0, 9] == 0.0))

        joints_norm, pos, shuttle, video_len = make_seq_len_same(int(target_len), joints_norm, pos, shuttle)
        self.assertEqual(len(joints_norm), target_len)
        self.assertEqual(len(pos), target_len)
        self.assertEqual(len(shuttle), target_len)
        self.assertLessEqual(video_len, target_len)

        human_pose = build_jnb_bone(joints_norm)
        self.assertEqual(human_pose.shape, (target_len, 2, 36, 2))

        batch = prepare_bst_batch(
            [
                {
                    "human_pose": human_pose,
                    "pos": pos,
                    "shuttle": shuttle,
                    "video_len": np.asarray(video_len, dtype=np.int64),
                }
            ]
        )
        self.assertEqual(batch["human_pose"].shape, (1, target_len, 2, 36, 2))
        flattened = torch.from_numpy(batch["human_pose"]).view(1, target_len, 2, -1)
        self.assertEqual(tuple(flattened.shape), (1, target_len, 2, 72))

    @unittest.skipUnless(BST_WEIGHT.exists(), "local BST weight is not available")
    def test_strict_weight_load_and_forward(self) -> None:
        model = build_bst_model(BST_WEIGHT)
        self.assertEqual(model.bst_seq_len, 30)
        self.assertEqual(model.bst_n_classes, 25)
        self.assertEqual(model.bst_in_dim, 72)

        target_len = model.bst_seq_len
        human_pose = np.zeros((1, target_len, 2, 36, 2), dtype=np.float32)
        shuttle = np.zeros((1, target_len, 2), dtype=np.float32)
        pos = np.zeros((1, target_len, 2, 2), dtype=np.float32)
        video_len = np.asarray([target_len], dtype=np.int64)
        result = run_bst_inference(model, human_pose, shuttle, pos, video_len, "cpu")
        self.assertIn("pred_id", result)
        self.assertIn("pred_name", result)
        self.assertIn("confidence", result)
        self.assertIn("top5", result)
        self.assertEqual(tuple(result["logits"].shape), (1, 25))


if __name__ == "__main__":
    unittest.main()
