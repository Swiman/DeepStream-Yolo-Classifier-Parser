/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yoloPlugins.h"

namespace
{
  template <typename T>
  void write(char *&buffer, const T &val)
  {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
  }
  template <typename T>
  void read(const char *&buffer, T &val)
  {
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
  }
}

cudaError_t cudaYoloLayer(const void *input, void *scores, const uint &batchSize,
                          const uint64_t &outputSize, cudaStream_t stream);

YoloLayer::YoloLayer(const void *data, size_t length)
{
  const char *d = static_cast<const char *>(data);
  read(d, m_OutputSize);
};

YoloLayer::YoloLayer(const uint64_t &outputSize) : m_OutputSize(outputSize)
{
  assert(outputSize > 0);
};

nvinfer1::IPluginV2DynamicExt *
YoloLayer::clone() const noexcept
{
  return new YoloLayer(m_OutputSize);
}

size_t
YoloLayer::getSerializationSize() const noexcept
{
  return sizeof(m_OutputSize);
}

void YoloLayer::serialize(void *buffer) const noexcept
{
  char *d = static_cast<char *>(buffer);
  write(d, m_OutputSize);
}

nvinfer1::DimsExprs
YoloLayer::getOutputDimensions(INT index, const nvinfer1::DimsExprs *inputs, INT nbInputDims,
                               nvinfer1::IExprBuilder &exprBuilder) noexcept
{
  assert(index < 3);
  if (index == 0)
  {
    return nvinfer1::DimsExprs{3, {inputs->d[0], exprBuilder.constant(static_cast<int>(m_OutputSize)), exprBuilder.constant(4)}};
  }
  return nvinfer1::DimsExprs{3, {inputs->d[0], exprBuilder.constant(static_cast<int>(m_OutputSize)), exprBuilder.constant(1)}};
}

bool YoloLayer::supportsFormatCombination(INT pos, const nvinfer1::PluginTensorDesc *inOut, INT nbInputs, INT nbOutputs) noexcept
{
  return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

nvinfer1::DataType
YoloLayer::getOutputDataType(INT index, const nvinfer1::DataType *inputTypes, INT nbInputs) const noexcept
{
  assert(index < 3);
  return nvinfer1::DataType::kFLOAT;
}

void YoloLayer::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, INT nbInput,
                                const nvinfer1::DynamicPluginTensorDesc *out, INT nbOutput) noexcept
{
  assert(nbInput > 0);
  assert(in->desc.format == nvinfer1::PluginFormat::kLINEAR);
  assert(in->desc.dims.d != nullptr);
}

INT YoloLayer::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                       void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
  INT batchSize = inputDesc[0].dims.d[0];
  void *scores = outputs[0];
  cudaYoloLayer(inputs[0], outputs[0], batchSize, m_OutputSize, stream);
  return 0;
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
