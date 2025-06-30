
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NLLLossTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, mode);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 2, shape)

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NLLLoss, NLLLossTilingData)
}
