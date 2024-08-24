#include "utils/operator_utils.h"
#include "boost/iterator/zip_iterator.hpp"
#include "core/runtime.h"
#include "core/tensor.h"
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <cassert>
#include <cstdio>

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

  // =================================== 作业
  // ===================================
  // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
  // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
  // =================================== 作业
  // =================================== scale rule 1, A has the same shape as B
  if (A.size() == 0) {
    std::printf("rule 1: A.size() == 0\n");
    return B;
  } else if (B.size() == 0) {
    std::printf("rule 1: A.size() == 0\n");
    return A;
  }
  // rule 2, A, B has same shape dim, the dim is either common or 1
  if (A.size() == B.size()) {
    std::printf("rule 2: A.size() == B.size()\n");
    Shape res;
    for (int i = 0; i < (int)A.size(); i++) {
      if (A[i] != B[i] && A[i] != 1 && B[i] != 1) {
        assert(false && "A and B can't be broadcasted");
      }
      res.push_back(std::max(A[i], B[i]));
    }
    return res;
  }

  // rule 3
  std::printf("rule 3: A.size() != B.size()\n");
  Shape result(std::max(A.size(), B.size()), 1);
  for (int i = 0; i < (int)(result.size()); ++i) {
    int A_idx{(int)(A.size()) - i - 1};
    int B_idx{(int)(B.size()) - i - 1};
    if (A_idx >= 0 && B_idx >= 0) {
      assert(A.at(A_idx) == B.at(B_idx) || A.at(A_idx) == 1 ||
             B.at(B_idx) == 1 || "A and B can't be broadcasted");
      result.at(result.size() - i - 1) = std::max(A.at(A_idx), B.at(B_idx));
    } else if (A_idx >= 0) {
      result.at(result.size() - i - 1) = A.at(A_idx);
    } else {
      result.at(result.size() - i - 1) = B.at(B_idx);
    }
  }

  return result;
}

int get_real_axis(const int &axis, const int &rank) {
  IT_ASSERT(rank >= 1);
  IT_ASSERT(axis >= -rank && axis <= (rank - 1));
  int newAxis;
  if (axis < 0) {
    newAxis = rank + axis;
  } else {
    newAxis = axis;
  }
  return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
  Shape ans(shape.size());
  auto i = ans.rbegin();
  auto j = shape.rbegin(), ej = shape.rend();
  while (j != ej) {
    auto div = std::div(inputN, *j++);
    *i++ = div.rem;
    inputN = div.quot;
  }
  return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
  size_t ans = 0;
  Shape index(shapeIndex.size());
  IT_ASSERT(shapeIndex.size() == shape.size());
  IT_ASSERT(shape.size() == stride.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    index[i] = shapeIndex[i] % shape[i];
    ans += index[i] * stride[i];
  }
  return ans;
}

std::string device_to_str(Device device) {
  std::string deviceStr;
  switch (device) {
  case Device::CPU:
    return "CPU";
  default:
    IT_TODO_HALT();
  }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
  std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
  std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
  return deviceStr + ", " + opStr;
}

} // namespace infini
