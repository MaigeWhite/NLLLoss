#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(NLLLoss)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(target, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLoss);

}

#endif
