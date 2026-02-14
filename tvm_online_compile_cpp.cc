#include <tvm/driver/compile.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/node/serialization.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>

int main() {
  DLDataType dl_type = {kDLFloat, 32, 1};
  tvm::runtime::DataType dtype = tvm::runtime::DataType(dl_type);
  DLDeviceType dl_dev_type = kDLOpenCL;
  auto tgt = tvm::Target("opencl");
  auto tgt_host = tvm::Target("llvm");
  tgt = tvm::Target::WithHost(tgt, tgt_host);

  auto N = tvm::tir::Var("N", tvm::runtime::DataType::Int(64));
  auto M = tvm::tir::Var("M", tvm::runtime::DataType::Int(64));
  tvm::ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                              M.as<tvm::PrimExpr>().value(), 4, 5};
  tvm::ffi::Array<tvm::PrimExpr> tvm_shape_c = {1, 1, 4, 5};

  tvm::relax::TensorStructInfo ts_info =
      tvm::relax::TensorStructInfo(tvm::relax::ShapeExpr(tvm_shape), dtype);
  tvm::relax::TensorStructInfo ts_info_c =
      tvm::relax::TensorStructInfo(tvm::relax::ShapeExpr(tvm_shape_c), dtype);
  auto A = tvm::relax::Var("A", ts_info);
  auto B = tvm::relax::Var("B", ts_info);
  auto input_tensor_c = tvm::runtime::Tensor::Empty(tvm::ffi::Shape({1, 1, 4, 5}), dl_type,
                                                    tvm::Device({dl_dev_type, 0}), std::nullopt);

  auto C = tvm::relax::Var("C", ts_info_c);
  tvm::ffi::Array<tvm::relax::Var> tvm_args;
  tvm_args.push_back(A);
  tvm_args.push_back(B);
  tvm_args.push_back(C);

  const tvm::Op& add_op_ = tvm::Op::Get("relax.add");
  const tvm::Op& subtract_op_ = tvm::Op::Get("relax.subtract");

  tvm::relax::BlockBuilder ctx_ = tvm::relax::BlockBuilder::Create(std::nullopt);

  ctx_->BeginScope(tvm_args);

  ctx_->BeginDataflowBlock();
  auto result = ctx_->Emit(tvm::relax::Call(add_op_, {A, B}, tvm::Attrs(), {}));
  result = ctx_->Emit(tvm::relax::Call(subtract_op_, {result, C}, tvm::Attrs(), {}));
  auto result_out = ctx_->EmitOutput(result, "add_out");
  auto bind_block = ctx_->EndBlock();

  auto body = ctx_->Normalize(result_out);
  body = ctx_->Normalize(tvm::relax::SeqExpr({bind_block}, body));

  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> fattrs;
  fattrs.Set(tvm::attr::kGlobalSymbol, tvm::ffi::String("main"));
  auto func = tvm::relax::Function(tvm_args, body, std::nullopt, true, tvm::DictAttrs(fattrs));
  ctx_->EndScope();

  ctx_->AddFunction(func, "main");
  tvm::IRModule mod_ = ctx_->Finalize();
  LOG(WARNING) << "Mod:" << mod_;

  auto vm_mod =
      tvm::driver::Compile(mod_, tgt, tvm::ffi::String("gpu_generic"), tvm::ffi::String("generic"));

  auto vm_ex = vm_mod.as<tvm::runtime::vm::VMExecutable>();

  LOG(WARNING) << "As text:" << vm_ex->AsText();

  LOG(WARNING) << "About to Load Executable";

  auto vm = vm_ex->VMLoadExecutable();
  vm->GetFunction("vm_initialization")
      .value()(dl_dev_type, 0, tvm::runtime::memory::AllocatorType::kPooled, kDLCPU, 0,
               tvm::runtime::memory::AllocatorType::kPooled);

  auto input_tensor_a = tvm::runtime::Tensor::Empty(tvm::ffi::Shape({10, 30, 4, 5}), dl_type,
                                                    tvm::Device({dl_dev_type, 0}), std::nullopt);
  auto input_tensor_b = tvm::runtime::Tensor::Empty(tvm::ffi::Shape({10, 30, 4, 5}), dl_type,
                                                    tvm::Device({dl_dev_type, 0}), std::nullopt);
  std::vector<tvm::ffi::AnyView> packed_args = {tvm::ffi::String("main"), input_tensor_a,
                                                input_tensor_b, input_tensor_c};

  tvm::ffi::Any ret;

  LOG(WARNING) << "About to set_input";
  vm->GetFunction("set_input")
      .value()
      .CallPacked(tvm::ffi::PackedArgs(packed_args.data(), packed_args.size()), &ret);

  LOG(WARNING) << "About to Invoke";
  vm->GetFunction("invoke_stateful").value()("main");

  LOG(WARNING) << "About to Get Output";
  auto output = vm->GetFunction("get_output").value()("main").cast<tvm::runtime::Tensor>();

  LOG(WARNING) << "Result:" << output.shape();
}
