#!/bin/bash
echo "[Ascend910B1] Generating NLLLoss_1fcbaa80dce92b59e8af83b59108685e ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=nll_loss --input_param=/home/maijianqiang/NLLLoss/build_out/op_kernel/binary/ascend910b/gen/NLLLoss_1fcbaa80dce92b59e8af83b59108685e_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/NLLLoss_1fcbaa80dce92b59e8af83b59108685e.json ; then
  echo "$2/NLLLoss_1fcbaa80dce92b59e8af83b59108685e.json not generated!"
  exit 1
fi

if ! test -f $2/NLLLoss_1fcbaa80dce92b59e8af83b59108685e.o ; then
  echo "$2/NLLLoss_1fcbaa80dce92b59e8af83b59108685e.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating NLLLoss_1fcbaa80dce92b59e8af83b59108685e Done"
