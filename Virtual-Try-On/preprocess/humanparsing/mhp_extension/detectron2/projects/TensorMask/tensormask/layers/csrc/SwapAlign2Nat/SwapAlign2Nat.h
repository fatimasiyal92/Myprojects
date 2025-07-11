// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace tensormask {

#ifdef WITH_cuda
at::Tensor SwapAlign2Nat_forward_cuda(
    const at::Tensor& X,
    const int lambda_val,
    const float pad_val);

at::Tensor SwapAlign2Nat_backward_cuda(
    const at::Tensor& gY,
    const int lambda_val,
    const int batch_size,
    const int channel,
    const int height,
    const int width);
#endif

inline at::Tensor SwapAlign2Nat_forward(
    const at::Tensor& X,
    const int lambda_val,
    const float pad_val) {
  if (X.type().is_cuda()) {
#ifdef WITH_cuda
    return SwapAlign2Nat_forward_cuda(X, lambda_val, pad_val);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the cuda");
}

inline at::Tensor SwapAlign2Nat_backward(
    const at::Tensor& gY,
    const int lambda_val,
    const int batch_size,
    const int channel,
    const int height,
    const int width) {
  if (gY.type().is_cuda()) {
#ifdef WITH_cuda
    return SwapAlign2Nat_backward_cuda(
        gY, lambda_val, batch_size, channel, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the cuda");
}

} // namespace tensormask
