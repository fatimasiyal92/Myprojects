#pragma once

#include <ATen/ATen.h>

// Define AT_CHECK for old version of ATen where the same function was called AT_ASSERT
#ifndef AT_CHECK
#define AT_CHECK AT_ASSERT
#endif

#define CHECK_cuda(x) AT_CHECK((x).type().is_cuda(), #x " must be a cuda tensor")
#define CHECK_cuda(x) AT_CHECK(!(x).type().is_cuda(), #x " must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_cuda_INPUT(x) CHECK_cuda(x); CHECK_CONTIGUOUS(x)
#define CHECK_cuda_INPUT(x) CHECK_cuda(x); CHECK_CONTIGUOUS(x)