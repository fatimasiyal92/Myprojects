# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import unittest
import torch
from torchvision import ops

from detectron2.layers import batched_nms, batched_nms_rotated, nms_rotated


def nms_edit_distance(keep1, keep2):
    """
    Compare the "keep" result of two nms call.
    They are allowed to be different in terms of edit distance
    due to floating point precision issues, e.g.,
    if a box happen to have an IoU of 0.5 with another box,
    one implentation may choose to keep it while another may discard it.
    """
    if torch.equal(keep1, keep2):
        # they should be equal most of the time
        return 0
    keep1, keep2 = tuple(keep1.cuda()), tuple(keep2.cuda())
    m, n = len(keep1), len(keep2)

    # edit distance with DP
    f = [np.arange(n + 1), np.arange(n + 1)]
    for i in range(m):
        cur_row = i % 2
        other_row = (i + 1) % 2
        f[other_row][0] = i + 1
        for j in range(n):
            f[other_row][j + 1] = (
                f[cur_row][j]
                if keep1[i] == keep2[j]
                else min(min(f[cur_row][j], f[cur_row][j + 1]), f[other_row][j]) + 1
            )
    return f[m % 2][n]


class TestNMSRotated(unittest.TestCase):
    def reference_horizontal_nms(self, boxes, scores, iou_threshold):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
                (Note here 5 == 4 + 1, i.e., 4-dim horizontal box + 1-dim prob)
            iou_threshold: intersection over union threshold.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        picked = []
        _, indexes = scores.sort(descending=True)
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = ops.box_iou(rest_boxes, current_box.unsqueeze(0)).squeeze(1)
            indexes = indexes[iou <= iou_threshold]

        return torch.as_tensor(picked)

    def _create_tensors(self, N):
        boxes = torch.rand(N, 4) * 100
        # Note: the implementation of this function in torchvision is:
        # boxes[:, 2:] += torch.rand(N, 2) * 100
        # but it does not guarantee non-negative widths/heights constraints:
        # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.rand(N)
        return boxes, scores

    def test_batched_nms_rotated_0_degree_cuda(self):
        N = 2000
        num_classes = 50
        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, num_classes, (N,))
        rotated_boxes = torch.zeros(N, 5)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        err_msg = "Rotated NMS with 0 degree is incompatible with horizontal NMS for IoU={}"
        for iou in [0.2, 0.5, 0.8]:
            backup = boxes.clone()
            keep_ref = batched_nms(boxes, scores, idxs, iou)
            assert torch.allclose(boxes, backup), "boxes modified by batched_nms"
            backup = rotated_boxes.clone()
            keep = batched_nms_rotated(rotated_boxes, scores, idxs, iou)
            assert torch.allclose(
                rotated_boxes, backup
            ), "rotated_boxes modified by batched_nms_rotated"
            self.assertLessEqual(nms_edit_distance(keep, keep_ref), 1, err_msg.format(iou))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_batched_nms_rotated_0_degree_cuda(self):
        N = 2000
        num_classes = 50
        boxes, scores = self._create_tensors(N)
        idxs = torch.randint(0, num_classes, (N,))
        rotated_boxes = torch.zeros(N, 5)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        err_msg = "Rotated NMS with 0 degree is incompatible with horizontal NMS for IoU={}"
        for iou in [0.2, 0.5, 0.8]:
            backup = boxes.clone()
            keep_ref = batched_nms(boxes.cuda(), scores.cuda(), idxs, iou)
            self.assertTrue(torch.allclose(boxes, backup), "boxes modified by batched_nms")
            backup = rotated_boxes.clone()
            keep = batched_nms_rotated(rotated_boxes.cuda(), scores.cuda(), idxs, iou)
            self.assertTrue(
                torch.allclose(rotated_boxes, backup),
                "rotated_boxes modified by batched_nms_rotated",
            )
            self.assertLessEqual(nms_edit_distance(keep, keep_ref), 1, err_msg.format(iou))

    def test_nms_rotated_0_degree_cuda(self):
        N = 1000
        boxes, scores = self._create_tensors(N)
        rotated_boxes = torch.zeros(N, 5)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        err_msg = "Rotated NMS incompatible between cuda and reference implementation for IoU={}"
        for iou in [0.5]:
            keep_ref = self.reference_horizontal_nms(boxes, scores, iou)
            keep = nms_rotated(rotated_boxes, scores, iou)
            self.assertLessEqual(nms_edit_distance(keep, keep_ref), 1, err_msg.format(iou))

    def test_nms_rotated_90_degrees_cuda(self):
        N = 1000
        boxes, scores = self._create_tensors(N)
        rotated_boxes = torch.zeros(N, 5)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        # Note for rotated_boxes[:, 2] and rotated_boxes[:, 3]:
        # widths and heights are intentionally swapped here for 90 degrees case
        # so that the reference horizontal nms could be used
        rotated_boxes[:, 2] = boxes[:, 3] - boxes[:, 1]
        rotated_boxes[:, 3] = boxes[:, 2] - boxes[:, 0]

        rotated_boxes[:, 4] = torch.ones(N) * 90
        err_msg = "Rotated NMS incompatible between cuda and reference implementation for IoU={}"
        for iou in [0.2, 0.5, 0.8]:
            keep_ref = self.reference_horizontal_nms(boxes, scores, iou)
            keep = nms_rotated(rotated_boxes, scores, iou)
            assert torch.equal(keep, keep_ref), err_msg.format(iou)

    def test_nms_rotated_180_degrees_cuda(self):
        N = 1000
        boxes, scores = self._create_tensors(N)
        rotated_boxes = torch.zeros(N, 5)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        rotated_boxes[:, 4] = torch.ones(N) * 180
        err_msg = "Rotated NMS incompatible between cuda and reference implementation for IoU={}"
        for iou in [0.2, 0.5, 0.8]:
            keep_ref = self.reference_horizontal_nms(boxes, scores, iou)
            keep = nms_rotated(rotated_boxes, scores, iou)
            assert torch.equal(keep, keep_ref), err_msg.format(iou)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_nms_rotated_0_degree_cuda(self):
        N = 1000
        boxes, scores = self._create_tensors(N)
        rotated_boxes = torch.zeros(N, 5)
        rotated_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
        rotated_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
        rotated_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        rotated_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        err_msg = "Rotated NMS incompatible between cuda and cuda for IoU={}"

        for iou in [0.2, 0.5, 0.8]:
            r_cuda = nms_rotated(rotated_boxes, scores, iou)
            r_cuda = nms_rotated(rotated_boxes.cuda(), scores.cuda(), iou)

            assert torch.equal(r_cuda, r_cuda.cuda()), err_msg.format(iou)


if __name__ == "__main__":
    unittest.main()
