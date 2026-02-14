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

tvm::IRModule apply_pass(std::string pass_name, tvm::IRModule& mod) {
  auto pass = tvm::ffi::Function::GetGlobal(pass_name);
  if (pass) {
    LOG(WARNING) << "Pass Applying:" << pass_name;
    tvm::transform::Pass pass_handler = (*pass)().cast<tvm::transform::Pass>();
    return pass_handler(mod);
  } else {
    LOG(FATAL) << "Can't find Pass:" << pass_name;
  }
}

std::pair<tvm::IRModule, tvm::ffi::Map<tvm::Target, tvm::IRModule>> SplitHostDeviceMods(
    tvm::IRModule& mod) {
  tvm::Target target;
  tvm::IRModule host_mod;
  tvm::ffi::Map<tvm::Target, tvm::IRModule> device_mod_dict;
  bool is_host = true;

  std::function<bool(const tvm::tir::PrimFunc&)> is_host_func =
      [&](const tvm::tir::PrimFunc& func) -> bool {
    tvm::Target tgt = func->GetAttr<tvm::Target>("target", tvm::Target("llvm")).value();
    if (tgt->kind->name == "llvm" || tgt->kind->name == "c") {
      return is_host ? true : false;
    } else {
      return is_host ? false : true;
    }
  };

  auto pass = tvm::ffi::Function::GetGlobal("tir.transform.Filter");
  host_mod = (*pass)(tvm::ffi::TypedFunction<bool(tvm::tir::PrimFunc)>(is_host_func))
                 .cast<tvm::transform::Pass>()(mod);
  is_host = false;
  auto device_mod = (*pass)(tvm::ffi::TypedFunction<bool(tvm::tir::PrimFunc)>(is_host_func))
                        .cast<tvm::transform::Pass>()(mod);

  tvm::ffi::Map<tvm::ffi::String, tvm::Target> target_str2target;
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc>> device_func_dict;
  for (auto it : device_mod->functions) {
    tvm::Target tgt = it.second->GetAttr<tvm::Target>("target", tvm::Target()).value();
    target_str2target.Set(tgt->str(), tgt);
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> dev_funcs;
    if (device_func_dict.find(tgt->str()) != device_func_dict.end()) {
      dev_funcs = device_func_dict[tgt->str()];
    }
    dev_funcs.Set(it.first, it.second);
    device_func_dict.Set(tgt->str(), dev_funcs);
  }

  for (auto it : device_func_dict) {
    auto tgt = tvm::Target(it.first);

    auto d_mod =
        tvm::IRModule(it.second, tvm::SourceMap({}), device_mod->attrs,
                      tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>({}));
    device_mod_dict.Set(tgt, d_mod);
  }

  return std::pair<tvm::IRModule, tvm::ffi::Map<tvm::Target, tvm::IRModule>>(
      {host_mod, device_mod_dict});
}

int main() {
  DLDataType dl_type = {kDLFloat, 32, 1};
  tvm::runtime::DataType dtype = tvm::runtime::DataType(dl_type);
  DLDeviceType dl_dev_type = kDLCPU;
  auto tgt = tvm::Target("llvm");
  auto tgt_host = tvm::Target("llvm");
  tgt = tvm::Target::WithHost(tgt, tgt_host);

  auto N = tvm::tir::Var("N", tvm::runtime::DataType::Int(64));
  auto M = tvm::tir::Var("M", tvm::runtime::DataType::Int(64));
  tvm::ffi::Array<tvm::PrimExpr> tvm_shape = {N.as<tvm::PrimExpr>().value(),
                                              M.as<tvm::PrimExpr>().value(), 4, 5};
  // tvm::ffi::Array<tvm::PrimExpr> tvm_shape = {2, 3, 4, 5};
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

  // Relax Pipeline
  mod_ = apply_pass("relax.transform.Normalize", mod_);
  mod_ = apply_pass("relax.transform.CanonicalizeBindings", mod_);
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> cmap;
  tvm::ffi::Array<tvm::ffi::String> skip_ops;
  auto pass = tvm::ffi::Function::GetGlobal("relax.transform.LegalizeOps");
  if (pass) {
    LOG(WARNING) << "Pass Applying: relax.transform.LegalizeOps";
    tvm::transform::Pass pass_handler = (*pass)(cmap, skip_ops, true).cast<tvm::transform::Pass>();
    mod_ = pass_handler(mod_);
  } else {
    LOG(FATAL) << "Can't find Pass: relax.transform.LegalizeOps";
  }
  pass = tvm::ffi::Function::GetGlobal("relax.transform.DeadCodeElimination");
  tvm::transform::Pass pass_h =
      (*pass)(tvm::ffi::Array<tvm::ffi::String>({})).cast<tvm::transform::Pass>();
  mod_ = pass_h(mod_);

  mod_ = apply_pass("relax.transform.AnnotateTIROpPattern", mod_);
  mod_ = apply_pass("relax.transform.FoldConstant", mod_);
  pass = tvm::ffi::Function::GetGlobal("relax.transform.FuseOps");
  pass_h = (*pass)(-1).cast<tvm::transform::Pass>();
  mod_ = pass_h(mod_);
  mod_ = apply_pass("relax.transform.FuseTIR", mod_);

  pass = tvm::ffi::Function::GetGlobal("relax.transform.DeadCodeElimination");
  pass_h = (*pass)(tvm::ffi::Array<tvm::ffi::String>({})).cast<tvm::transform::Pass>();
  mod_ = pass_h(mod_);

  mod_ = apply_pass("relax.transform.AttachAttrLayoutFreeBuffers", mod_);

  LOG(WARNING) << "About to meta tune";
  pass = tvm::ffi::Function::GetGlobal("target.TargetEnterScope");
  (*pass)(tgt);

  /*
  pass = tvm::ffi::Function::GetGlobal("relax.transform.MetaScheduleTuneIRMod");
  pass_h = (*pass)(tvm::ffi::Map<tvm::ffi::String, tvm::runtime::Tensor>({}), "./meta_tune", 3, 1,
  tvm::ffi::Optional<tvm::ffi::Array<tvm::ffi::String>>()).cast<tvm::transform::Pass>(); mod_ =
  pass_h(mod_);

  pass = tvm::ffi::Function::GetGlobal("relax.transform.MetaScheduleApplyDatabase");
  pass_h = (*pass)("./meta_tune", true).cast<tvm::transform::Pass>();
  mod_ = pass_h(mod_);
  */

  // mod_ = apply_pass("relax.transform.SplitLayoutRewritePreproc", mod_);
  // mod_ = apply_pass("relax.transform.LiftTransformParams", mod_);
  // mod_ = apply_pass("relax.transform.FoldConstant", mod_);

  pass = tvm::ffi::Function::GetGlobal("relax.transform.ApplyDlightSchedule");

  pass_h = (*pass)(tvm::ffi::Array<tvm::ffi::String>(
                       {"dl.adreno.Conv2D", "dl.adreno.LayoutTransform", "dl.adreno.Pool2D",
                        "dl.gpu.Reduction", "dl.gpu.GeneralReduction", "dl.gpu.Fallback"}))
               .cast<tvm::transform::Pass>();
  mod_ = pass_h(mod_);

  mod_ = apply_pass("relax.transform.RewriteDataflowReshape", mod_);
  mod_ = apply_pass("relax.transform.ToNonDataflow", mod_);
  mod_ = apply_pass("relax.transform.RemovePurityChecking", mod_);
  mod_ = apply_pass("relax.transform.CallTIRRewrite", mod_);
  mod_ = apply_pass("relax.transform.StaticPlanBlockMemory", mod_);
  mod_ = apply_pass("relax.transform.LowerAllocTensor", mod_);
  mod_ = apply_pass("relax.transform.KillAfterLastUse", mod_);
  mod_ = apply_pass("relax.transform.LowerRuntimeBuiltin", mod_);
  mod_ = apply_pass("relax.transform.ComputePrimValue", mod_);
  pass = tvm::ffi::Function::GetGlobal("relax.transform.VMShapeLower");
  pass_h = (*pass)(true).cast<tvm::transform::Pass>();
  mod_ = pass_h(mod_);
  mod_ = apply_pass("relax.transform.AttachGlobalSymbol", mod_);
  LOG(WARNING) << "Mod:" << mod_;

  // Relax Build

  auto gfunc = tvm::ffi::Function::GetGlobal("relax.ExecBuilderCreate");
  tvm::relax::ExecBuilder ex_builder = (*gfunc)().cast<tvm::relax::ExecBuilder>();

  gfunc = tvm::ffi::Function::GetGlobal("relax.VMCodeGen");
  LOG(WARNING) << "About to Build:" << mod_;
  auto b_mod = (*gfunc)(ex_builder, mod_).cast<tvm::IRModule>();
  LOG(WARNING) << "B Mod:" << b_mod;

  std::function<tvm::IRModule(tvm::IRModule&)> _filter_tir =
      [&](tvm::IRModule& mod) -> tvm::IRModule {
    tvm::ffi::Map<tvm::GlobalVar, tvm::BaseFunc> tir_funcs;
    for (auto func : mod->functions) {
      if (func.second.as<tvm::tir::PrimFunc>()) {
        tir_funcs.Set(func.first, func.second);
      }
    }
    return tvm::IRModule(tir_funcs, tvm::SourceMap({}), mod->attrs,
                         tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>({}));
  };

  auto tir_mod = _filter_tir(b_mod);

  // TIR Build
  // auto lib = TIRBuild(tir_mod, target)
  pass = tvm::ffi::Function::GetGlobal("tir.transform.BindTarget");
  pass_h = (*pass)(tgt).cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  // TIR Pipeline
  tir_mod = apply_pass("tir.transform.CanonicalizeLoop", tir_mod);
  tir_mod = apply_pass("tir.transform.LowerCrossThreadReduction", tir_mod);
  tir_mod = apply_pass("tir.transform.LowerInitBlock", tir_mod);
  tir_mod = apply_pass("tir.transform.PlanAndUpdateBufferAllocationLocation", tir_mod);
  tir_mod = apply_pass("tir.transform.ConvertBlocksToOpaque", tir_mod);
  tir_mod = apply_pass("tir.transform.LiftThreadBinding", tir_mod);
  tir_mod = apply_pass("tir.transform.ManifestSharedMemoryLocalStage", tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.CompactBufferAllocation");
  pass_h = (*pass)(true).cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  tir_mod = apply_pass("tir.transform.LowerAutoCopy", tir_mod);
  tir_mod = apply_pass("tir.transform.UnifyThreadBinding", tir_mod);
  tir_mod = apply_pass("tir.transform.LowerMatchBuffer", tir_mod);
  tir_mod = apply_pass("tir.transform.Simplify", tir_mod);
  tir_mod = apply_pass("tir.transform.InjectPermutedLayout", tir_mod);
  tir_mod = apply_pass("tir.transform.AnnotateIrregularLoop", tir_mod);
  tir_mod = apply_pass("tir.transform.InjectSoftwarePipeline", tir_mod);
  tir_mod = apply_pass("tir.transform.TransformMmaBufferLayout", tir_mod);
  tir_mod = apply_pass("tir.transform.LowerOpaqueBlock", tir_mod);
  tir_mod = apply_pass("tir.transform.FlattenBuffer", tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.NarrowDataType");
  pass_h = (*pass)(32).cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  tir_mod = apply_pass("tir.transform.LoopPartition", tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.VectorizeLoop");
  pass_h = (*pass)(true).cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  tir_mod = apply_pass("tir.transform.InjectVirtualThread", tir_mod);
  tir_mod = apply_pass("tir.transform.InjectDoubleBuffer", tir_mod);

  tir_mod = apply_pass("tir.transform.StorageRewrite", tir_mod);
  tir_mod = apply_pass("tir.transform.HoistIfThenElse", tir_mod);
  tir_mod = apply_pass("tir.transform.UnrollLoop", tir_mod);
  tir_mod = apply_pass("tir.transform.RenormalizeSplitPattern", tir_mod);
  tir_mod = apply_pass("tir.transform.Simplify", tir_mod);
  tir_mod = apply_pass("tir.transform.RemoveNoOp", tir_mod);
  tir_mod = apply_pass("tir.transform.RewriteUnsafeSelect", tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.CommonSubexprElimTIR");
  pass_h = (*pass)(true, false).cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  tir_mod = apply_pass("tir.transform.VerifyMemory", tir_mod);
  tir_mod = apply_pass("tir.transform.AnnotateEntryFunc", tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.ThreadSync");
  pass_h = (*pass)("shared").cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.ThreadSync");
  pass_h = (*pass)("shared.dyn").cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  pass = tvm::ffi::Function::GetGlobal("tir.transform.ThreadSync");
  pass_h = (*pass)("warp").cast<tvm::transform::Pass>();
  tir_mod = pass_h(tir_mod);

  tir_mod = apply_pass("tir.transform.InferFragment", tir_mod);
  tir_mod = apply_pass("tir.transform.LowerThreadAllreduce", tir_mod);
  tir_mod = apply_pass("tir.transform.AnnotateDeviceRegions", tir_mod);
  tir_mod = apply_pass("tir.transform.SplitHostDevice", tir_mod);
  tir_mod = apply_pass("tir.transform.MergeSharedMemoryAllocations", tir_mod);
  tir_mod = apply_pass("tir.transform.MakePackedAPI", tir_mod);
  tir_mod = apply_pass("tir.transform.LowerDeviceKernelLaunch", tir_mod);

  LOG(WARNING) << "TIR Mod:" << tir_mod;
  auto [host_mod, device_mod_dict] = SplitHostDeviceMods(tir_mod);

  // Host finalize
  host_mod = apply_pass("tir.transform.LowerTVMBuiltin", host_mod);
  host_mod = apply_pass("tir.transform.LowerCustomDatatypes", host_mod);
  host_mod = apply_pass("tir.transform.LowerIntrin", host_mod);
  host_mod = apply_pass("tir.transform.LowerDeviceStorageAccessInfo", host_mod);
  host_mod = apply_pass("tir.transform.CombineContextCall", host_mod);

  // Device finalize
  for (auto [tgt, dmod] : device_mod_dict) {
    dmod = apply_pass("tir.transform.LowerWarpMemory", dmod);
    dmod = apply_pass("tir.transform.Simplify", dmod);
    dmod = apply_pass("tir.transform.LowerCustomDatatypes", dmod);
    dmod = apply_pass("tir.transform.LowerDeviceStorageAccessInfo", dmod);
    dmod = apply_pass("tir.transform.LowerIntrin", dmod);
    device_mod_dict.Set(tgt, dmod);
  }

  // TIR To Runtime
  auto mhost_all =
      tvm::IRModule({}, tvm::SourceMap({}), host_mod->attrs,
                    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Array<tvm::GlobalInfo>>({}));
  mhost_all->Update(host_mod);
  tvm::ffi::Array<tvm::ffi::Module> runtime_mods;

  for (auto [tgt, d_mod] : device_mod_dict) {
    if (d_mod->functions.size() != 0) {
      auto codegen_f =
          tvm::ffi::Function::GetGlobal(std::string("target.build.") + tgt->kind->name);
      runtime_mods.push_back((*codegen_f)(d_mod, tgt).cast<tvm::ffi::Module>());
    }
  }

  auto codegen_host =
      tvm::ffi::Function::GetGlobal(std::string("target.build.") + tgt_host->kind->name);
  auto m_host = (*codegen_host)(host_mod, tgt_host).cast<tvm::ffi::Module>();

  for (auto r_mod : runtime_mods) {
    m_host->ImportModule(r_mod);
  }

  /*tvm::ffi::Module VMLink(ExecBuilder builder, Target target, ffi::Optional<ffi::Module> lib,
                   ffi::Array<ffi::Module> ext_libs,
                   ffi::Map<ffi::String, runtime::Tensor> params) {*/

  gfunc = tvm::ffi::Function::GetGlobal("relax.VMLink");
  auto vm_mod = (*gfunc)(ex_builder, tgt, m_host, tvm::ffi::Array<tvm::ffi::Module>({}),
                         tvm::ffi::Map<tvm::ffi::String, tvm::runtime::Tensor>({}))
                    .cast<tvm::ffi::Module>();
  auto vm_ex = vm_mod.as<tvm::runtime::vm::VMExecutable>();

  LOG(WARNING) << "As text:" << vm_ex->AsText();

  auto vm = vm_ex->VMLoadExecutable();
  vm->GetFunction("vm_initialization")
      .value()(dl_dev_type, 0, tvm::runtime::memory::AllocatorType::kPooled);

  //  vm.set_input("main", *inputs)
  auto input_tensor_a = tvm::runtime::Tensor::Empty(tvm::ffi::Shape({10, 30, 4, 5}), dl_type,
                                                    tvm::Device({dl_dev_type, 0}), std::nullopt);
  auto input_tensor_b = tvm::runtime::Tensor::Empty(tvm::ffi::Shape({10, 30, 4, 5}), dl_type,
                                                    tvm::Device({dl_dev_type, 0}), std::nullopt);
  std::vector<tvm::ffi::AnyView> packed_args = {tvm::ffi::String("main"), input_tensor_a,
                                                input_tensor_b, input_tensor_c};
  // std::vector<tvm::ffi::AnyView> packed_args = {tvm::ffi::String("main"), input_tensor_a,
  //                                              input_tensor_b};

  tvm::ffi::Any ret;

  // vm->GetFunction("set_input").value()(main_args);
  vm->GetFunction("set_input")
      .value()
      .CallPacked(tvm::ffi::PackedArgs(packed_args.data(), packed_args.size()), &ret);
  vm->GetFunction("invoke_stateful").value()("main");

  auto output = vm->GetFunction("get_output").value()("main").cast<tvm::runtime::Tensor>();
  LOG(WARNING) << "Result:" << output.shape();

  LOG(WARNING) << " -- END --";
}
