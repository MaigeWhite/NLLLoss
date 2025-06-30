#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X>class KernelAdd {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAdd(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, int32_t *shape, int mode)
    {
        int32_t N = shape[0];
        int32_t C = shape[1];
        if (C == 65535){
            C = N;
            N = 1;
        }
        if constexpr (std::is_same_v<T, half>|std::is_same_v<T, float>){

            int32_t in_size=1;
            int32_t out_size=1;
            if(mode==1){
                out_size = N;
            }
        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, N*C);
        yGm.SetGlobalBuffer((__gm__ int32_t *)target, N);
        zGm.SetGlobalBuffer((__gm__ TYPE_X *)weight, C);
        oGm.SetGlobalBuffer((__gm__ TYPE_X *)y, out_size);
        float cur;
        float w;
        float sum=0.0;
        float al_w=0.0;

        for(int i=0; i<N; i++){
            cur = -(float)xGm.GetValue(i*C + yGm.GetValue(i));
            w = (float)zGm.GetValue(yGm.GetValue(i));
            sum += cur * w;
            al_w += w;
            if(mode==1){
                oGm.SetValue(i, (TYPE_X)(cur * w));
            }
        }
        if(mode==2){
            sum = sum/(al_w);
            oGm.SetValue(0, (TYPE_X)(sum));
        }
        if(mode==3){
            oGm.SetValue(0, (TYPE_X)(sum));
        }
    }
}
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<int32_t> yGm;
    AscendC::GlobalTensor<TYPE_X> oGm;
    AscendC::GlobalTensor<TYPE_X> zGm;
};


extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd<DTYPE_X> op;
    op.Init(x, target, weight, y, tiling_data.shape, tiling_data.mode);
    // op.Process();
    // TODO: user kernel impl
}
