// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#include <cuda_runtime_api.h>

namespace detectron2 {
int get_cudart_version() {
  return cudaRT_VERSION;
}
} // namespace detectron2
