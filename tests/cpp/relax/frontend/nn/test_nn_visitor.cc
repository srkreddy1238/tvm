/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tests/cpp/relax/frontend/nn/test_nn_visitor.cc
 * \brief C++ port of tests/python/relax/test_frontend_nn_visitor.py
 *
 * Ported tests
 * ────────────
 *   NamingBasic              – dotted names for a four-level nested Module.
 *   NamingModuleDict         – dotted names through ModuleDict → ModuleList.
 *   NamingModuleList         – dotted names through ModuleList → ModuleList.
 *   MutateModule             – visit_module replaces a sub-module.
 *   MutateModuleDict         – visit_module replaces an entry in a ModuleDict.
 *   MutateModuleList         – visit_module replaces an entry in a ModuleList.
 *   MutateEffect             – visit_effect replaces an Effect child.
 *   MutateParam              – visit_param replaces a Parameter.
 *   MutateRecursively        – visit_param recurses into nested Modules.
 *   DefaultVisitParam        – base visit_param returns node unchanged.
 *   DefaultVisitModule       – base visit_module returns node unchanged.
 *   NoopOnNonModule          – visit() on a non-module Any returns it unchanged.
 *   ModuleListNestedModuleDict – ModuleList containing ModuleDicts traversed.
 *   ParamDtypeUpgrade        – visit_param upgrades float16 → float32 everywhere.
 *
 * Design notes
 * ────────────
 * In Python, Module children live in __dict__.  In C++ they live in
 * NNModuleNode::attrs (a ffi::Map<String,Any>).  The Make* factory
 * functions populate attrs; for test modules we populate attrs directly.
 *
 * EffectNode is abstract in C++.  We use IOEffectModuleNode (a concrete
 * subclass) as a stand-in for the Python Effect stub classes.
 *
 * NNParameter is an ObjectRef wrapping ParameterNode.  We create them via
 * TensorNode::MakePlaceholder + NNParameter constructor.
 */

#include <gtest/gtest.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/object.h>

#include <map>
#include <string>
#include <vector>

#include "../../../../../src/relax/frontend/nn/core.h"
#include "../../../../../src/relax/frontend/nn/modules.h"
#include "../../../../../src/relax/frontend/nn/visitor.h"

namespace tvm {
namespace relax {
namespace frontend {
namespace nn {
namespace testing {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a scalar NNParameter with the given dtype (shape = [4]).
static NNParameter MakeParam4(const std::string& dtype) {
  ffi::Array<ffi::Any> shape{ffi::Any(int64_t(4))};
  TensorNode* raw = TensorNode::MakePlaceholder(shape, dtype, "param");
  Var v = raw->expr;
  delete raw;
  return NNParameter(v, std::nullopt, {});
}

// Build a (32, 128) NNParameter with the given dtype.
static NNParameter MakeParam32x128(const std::string& dtype) {
  ffi::Array<ffi::Any> shape{ffi::Any(int64_t(32)), ffi::Any(int64_t(128))};
  TensorNode* raw = TensorNode::MakePlaceholder(shape, dtype, "param");
  Var v = raw->expr;
  delete raw;
  return NNParameter(v, std::nullopt, {});
}

// Build a minimal NNModule with a single parameter child.
static NNModule LeafModule(const std::string& dtype) {
  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("param", ffi::Any(MakeParam4(dtype)));
  return NNModule(attrs);
}

// Return the dtype string of an NNParameter stored as ffi::Any.
static std::string ParamDtype(ffi::Any val) {
  auto opt = val.try_cast<NNParameter>();
  EXPECT_TRUE(opt.has_value()) << "Expected NNParameter";
  return std::string(opt.value()->GetDtype());
}

// ---------------------------------------------------------------------------
// Recorder mutator: collects (name, dtype) for every visited parameter.
// ---------------------------------------------------------------------------
class RecorderMutator : public Mutator {
 public:
  std::vector<std::pair<std::string, std::string>> seen;  // (name, dtype)

  ffi::Any visit_param(const std::string& name, NNParameter node) override {
    seen.emplace_back(name, std::string(node->GetDtype()));
    return ffi::Any(node);
  }
};

// ---------------------------------------------------------------------------
// NamingBasic
//
// Python equivalent:
//   class Module0: param0 (float64)
//   class Module1: mod0=Module0, param1 (float32)
//   class Module2: mod1=Module1, param2 (float16)
//   class Module3: mod2=Module2, param3 (float8)
//   mutator.visit("mod3", Module3())
//   # expected names:
//   #   mod3.param3, mod3.mod2.param2,
//   #   mod3.mod2.mod1.param1, mod3.mod2.mod1.mod0.param0
// ---------------------------------------------------------------------------
TEST(NNVisitor, NamingBasic) {
  // Build the four-level hierarchy bottom-up.
  NNModule mod0(ffi::Map<ffi::String, ffi::Any>{
      {ffi::String("param0"), ffi::Any(MakeParam32x128("float64"))}});

  ffi::Map<ffi::String, ffi::Any> attrs1;
  attrs1.Set("mod0", ffi::Any(mod0));
  attrs1.Set("param1", ffi::Any(MakeParam32x128("float32")));
  NNModule mod1(attrs1);

  ffi::Map<ffi::String, ffi::Any> attrs2;
  attrs2.Set("mod1", ffi::Any(mod1));
  attrs2.Set("param2", ffi::Any(MakeParam32x128("float16")));
  NNModule mod2(attrs2);

  ffi::Map<ffi::String, ffi::Any> attrs3;
  attrs3.Set("mod2", ffi::Any(mod2));
  attrs3.Set("param3", ffi::Any(MakeParam32x128("float8")));
  NNModule mod3(attrs3);

  std::map<std::string, std::string> expected{
      {"float8", "mod3.param3"},
      {"float16", "mod3.mod2.param2"},
      {"float32", "mod3.mod2.mod1.param1"},
      {"float64", "mod3.mod2.mod1.mod0.param0"},
  };

  class CheckMutator : public Mutator {
   public:
    const std::map<std::string, std::string>& expected;
    explicit CheckMutator(const std::map<std::string, std::string>& e) : expected(e) {}

    ffi::Any visit_param(const std::string& name, NNParameter node) override {
      std::string dtype = std::string(node->GetDtype());
      auto it = expected.find(dtype);
      EXPECT_NE(it, expected.end()) << "Unexpected dtype: " << dtype;
      if (it != expected.end()) {
        EXPECT_EQ(name, it->second) << "dtype=" << dtype;
      }
      return ffi::Any(node);
    }
  };

  CheckMutator mutator(expected);
  mutator.visit("mod3", ffi::Any(mod3));
}

// ---------------------------------------------------------------------------
// NamingModuleDict
//
// Python equivalent:
//   mod_dict = ModuleDict({
//       "k0": ModuleList([Leaf("float64"), Leaf("float32")]),
//       "k1": ModuleList([Leaf("float16"), Leaf("float8")]),
//   })
//   mutator.visit("mod_dict", mod_dict)
//   # expected: mod_dict.k0.0.param, mod_dict.k0.1.param,
//   #           mod_dict.k1.0.param, mod_dict.k1.1.param
// ---------------------------------------------------------------------------
TEST(NNVisitor, NamingModuleDict) {
  ModuleList list0(ffi::Array<ffi::Any>{ffi::Any(LeafModule("float64")),
                                        ffi::Any(LeafModule("float32"))});
  ModuleList list1(ffi::Array<ffi::Any>{ffi::Any(LeafModule("float16")),
                                        ffi::Any(LeafModule("float8"))});

  ffi::Map<ffi::String, ffi::Any> dict_modules;
  dict_modules.Set("k0", ffi::Any(list0));
  dict_modules.Set("k1", ffi::Any(list1));
  ModuleDict md(dict_modules);

  std::map<std::string, std::string> expected{
      {"float64", "mod_dict.k0.0.param"},
      {"float32", "mod_dict.k0.1.param"},
      {"float16", "mod_dict.k1.0.param"},
      {"float8", "mod_dict.k1.1.param"},
  };

  class CheckMutator : public Mutator {
   public:
    const std::map<std::string, std::string>& expected;
    explicit CheckMutator(const std::map<std::string, std::string>& e) : expected(e) {}

    ffi::Any visit_param(const std::string& name, NNParameter node) override {
      std::string dtype = std::string(node->GetDtype());
      auto it = expected.find(dtype);
      EXPECT_NE(it, expected.end()) << "Unexpected dtype: " << dtype;
      if (it != expected.end()) { EXPECT_EQ(name, it->second) << "dtype=" << dtype; }
      return ffi::Any(node);
    }
  };

  CheckMutator mutator(expected);
  mutator.visit("mod_dict", ffi::Any(md));
}

// ---------------------------------------------------------------------------
// NamingModuleList
//
// Python equivalent:
//   mod_list = ModuleList([
//       ModuleList([Leaf("float64"), Leaf("float32")]),
//       ModuleList([Leaf("float16"), Leaf("float8")]),
//   ])
//   mutator.visit("mod_list", mod_list)
// ---------------------------------------------------------------------------
TEST(NNVisitor, NamingModuleList) {
  ModuleList inner0(ffi::Array<ffi::Any>{ffi::Any(LeafModule("float64")),
                                         ffi::Any(LeafModule("float32"))});
  ModuleList inner1(ffi::Array<ffi::Any>{ffi::Any(LeafModule("float16")),
                                         ffi::Any(LeafModule("float8"))});
  ModuleList outer(ffi::Array<ffi::Any>{ffi::Any(inner0), ffi::Any(inner1)});

  std::map<std::string, std::string> expected{
      {"float64", "mod_list.0.0.param"},
      {"float32", "mod_list.0.1.param"},
      {"float16", "mod_list.1.0.param"},
      {"float8", "mod_list.1.1.param"},
  };

  class CheckMutator : public Mutator {
   public:
    const std::map<std::string, std::string>& expected;
    explicit CheckMutator(const std::map<std::string, std::string>& e) : expected(e) {}

    ffi::Any visit_param(const std::string& name, NNParameter node) override {
      std::string dtype = std::string(node->GetDtype());
      auto it = expected.find(dtype);
      EXPECT_NE(it, expected.end()) << "Unexpected dtype: " << dtype;
      if (it != expected.end()) { EXPECT_EQ(name, it->second) << "dtype=" << dtype; }
      return ffi::Any(node);
    }
  };

  CheckMutator mutator(expected);
  mutator.visit("mod_list", ffi::Any(outer));
}

// ---------------------------------------------------------------------------
// MutateModule
//
// Python equivalent:
//   class Parent: mod = Sub1()
//   mutator replaces Sub1 with Sub2 via visit_module
//   assert isinstance(parent.mod, Sub2)
//
// In C++ we use two distinct NNModule instances and distinguish them by a
// sentinel attribute ("kind" = "sub1" vs "sub2").
// ---------------------------------------------------------------------------
TEST(NNVisitor, MutateModule) {
  // Build a "Sub1" module: attrs = {kind: "sub1"}
  auto MakeSub = [](const std::string& kind) -> NNModule {
    ffi::Map<ffi::String, ffi::Any> a;
    a.Set("kind", ffi::Any(ffi::String(kind)));
    return NNModule(a);
  };

  NNModule sub1 = MakeSub("sub1");
  ffi::Map<ffi::String, ffi::Any> parent_attrs;
  parent_attrs.Set("mod", ffi::Any(sub1));
  NNModule parent(parent_attrs);

  // Verify initial state.
  auto get_kind = [](NNModule m) -> std::string {
    auto it = m->attrs.find("mod");
    EXPECT_NE(it, m->attrs.end());
    auto opt_ref = (*it).second.try_cast<NNModule>();
    EXPECT_TRUE(opt_ref.has_value());
    auto kind_it = opt_ref.value()->attrs.find("kind");
    EXPECT_NE(kind_it, opt_ref.value()->attrs.end());
    return std::string((*kind_it).second.cast<ffi::String>());
  };
  EXPECT_EQ(get_kind(parent), "sub1");

  // Mutator: replace "sub1" with "sub2".
  class ReplaceSub1 : public Mutator {
   public:
    ffi::Any visit_module(const std::string& /*name*/, runtime::ObjectRef node) override {
      if (const auto* m = node.as<NNModuleNode>()) {
        auto it = m->attrs.find("kind");
        if (it != m->attrs.end()) {
          auto opt = (*it).second.try_cast<ffi::String>();
          if (opt.has_value() && std::string(opt.value()) == "sub1") {
            ffi::Map<ffi::String, ffi::Any> a;
            a.Set("kind", ffi::Any(ffi::String("sub2")));
            return ffi::Any(NNModule(a));
          }
        }
      }
      return visit(/*name=*/"", ffi::Any(node));
    }
  };

  ReplaceSub1 mutator;
  ffi::Any result = mutator.visit("", ffi::Any(parent));
  NNModule result_mod = result.cast<NNModule>();
  EXPECT_EQ(get_kind(result_mod), "sub2");
}

// ---------------------------------------------------------------------------
// MutateModuleDict
//
// Python equivalent:
//   md = ModuleDict({"k0": M1(), "k1": M2(), "k2": M3()})
//   mutator replaces M3 with M1 via visit_module
//   assert isinstance(md["k2"], M1)
// ---------------------------------------------------------------------------
TEST(NNVisitor, MutateModuleDict) {
  auto MakeTagged = [](const std::string& tag) -> NNModule {
    ffi::Map<ffi::String, ffi::Any> a;
    a.Set("tag", ffi::Any(ffi::String(tag)));
    return NNModule(a);
  };

  auto GetTag = [](ffi::Any val) -> std::string {
    auto opt = val.try_cast<NNModule>();
    EXPECT_TRUE(opt.has_value());
    auto it = opt.value()->attrs.find("tag");
    EXPECT_NE(it, opt.value()->attrs.end());
    return std::string((*it).second.cast<ffi::String>());
  };

  ffi::Map<ffi::String, ffi::Any> dict_mods;
  dict_mods.Set("k0", ffi::Any(MakeTagged("m1")));
  dict_mods.Set("k1", ffi::Any(MakeTagged("m2")));
  dict_mods.Set("k2", ffi::Any(MakeTagged("m3")));
  ModuleDict md(dict_mods);

  EXPECT_EQ(GetTag(md->modules.at("k0")), "m1");
  EXPECT_EQ(GetTag(md->modules.at("k1")), "m2");
  EXPECT_EQ(GetTag(md->modules.at("k2")), "m3");

  class ReplaceM3 : public Mutator {
   public:
    ffi::Any visit_module(const std::string& name, runtime::ObjectRef node) override {
      if (const auto* m = node.as<NNModuleNode>()) {
        auto it = m->attrs.find("tag");
        if (it != m->attrs.end()) {
          auto opt = (*it).second.try_cast<ffi::String>();
          if (opt.has_value() && std::string(opt.value()) == "m3") {
            ffi::Map<ffi::String, ffi::Any> a;
            a.Set("tag", ffi::Any(ffi::String("m1")));
            return ffi::Any(NNModule(a));
          }
        }
      }
      return visit(name, ffi::Any(node));
    }
  };

  ReplaceM3 mutator;
  ffi::Any result = mutator.visit("", ffi::Any(md));
  ModuleDict result_md = result.cast<ModuleDict>();

  EXPECT_EQ(GetTag(result_md->modules.at("k0")), "m1");
  EXPECT_EQ(GetTag(result_md->modules.at("k1")), "m2");
  EXPECT_EQ(GetTag(result_md->modules.at("k2")), "m1");  // replaced
}

// ---------------------------------------------------------------------------
// MutateModuleList
//
// Python equivalent:
//   ml = ModuleList([M1(), M2(), M3()])
//   mutator replaces M3 with M1 via visit_module
//   assert isinstance(ml[2], M1)
// ---------------------------------------------------------------------------
TEST(NNVisitor, MutateModuleList) {
  auto MakeTagged = [](const std::string& tag) -> NNModule {
    ffi::Map<ffi::String, ffi::Any> a;
    a.Set("tag", ffi::Any(ffi::String(tag)));
    return NNModule(a);
  };

  auto GetTag = [](ffi::Any val) -> std::string {
    auto opt = val.try_cast<NNModule>();
    EXPECT_TRUE(opt.has_value());
    auto it = opt.value()->attrs.find("tag");
    EXPECT_NE(it, opt.value()->attrs.end());
    return std::string((*it).second.cast<ffi::String>());
  };

  ModuleList ml(ffi::Array<ffi::Any>{ffi::Any(MakeTagged("m1")), ffi::Any(MakeTagged("m2")),
                                      ffi::Any(MakeTagged("m3"))});

  EXPECT_EQ(GetTag(ml->modules[0]), "m1");
  EXPECT_EQ(GetTag(ml->modules[1]), "m2");
  EXPECT_EQ(GetTag(ml->modules[2]), "m3");

  class ReplaceM3 : public Mutator {
   public:
    ffi::Any visit_module(const std::string& name, runtime::ObjectRef node) override {
      if (const auto* m = node.as<NNModuleNode>()) {
        auto it = m->attrs.find("tag");
        if (it != m->attrs.end()) {
          auto opt = (*it).second.try_cast<ffi::String>();
          if (opt.has_value() && std::string(opt.value()) == "m3") {
            ffi::Map<ffi::String, ffi::Any> a;
            a.Set("tag", ffi::Any(ffi::String("m1")));
            return ffi::Any(NNModule(a));
          }
        }
      }
      return visit(name, ffi::Any(node));
    }
  };

  ReplaceM3 mutator;
  ffi::Any result = mutator.visit("", ffi::Any(ml));
  ModuleList result_ml = result.cast<ModuleList>();

  EXPECT_EQ(GetTag(result_ml->modules[0]), "m1");
  EXPECT_EQ(GetTag(result_ml->modules[1]), "m2");
  EXPECT_EQ(GetTag(result_ml->modules[2]), "m1");  // replaced
}

// ---------------------------------------------------------------------------
// MutateEffect
//
// Python equivalent:
//   class Parent: effect = IOEffect()
//   mutator replaces it with a second IOEffect via visit_effect
//
// We use IOEffectModuleNode as the concrete Effect type.  We distinguish
// the two instances by a sentinel attribute "id".
// ---------------------------------------------------------------------------
TEST(NNVisitor, MutateEffect) {
  // Build an IOEffect with a sentinel "id" attribute.
  auto MakeEffect = [](const std::string& id) -> IOEffectModule {
    IOEffectModule eff;
    eff->attrs.Set("id", ffi::Any(ffi::String(id)));
    return eff;
  };

  IOEffectModule eff1 = MakeEffect("eff1");
  ffi::Map<ffi::String, ffi::Any> parent_attrs;
  parent_attrs.Set("effect", ffi::Any(eff1));
  NNModule parent(parent_attrs);

  auto GetEffId = [](NNModule m) -> std::string {
    auto it = m->attrs.find("effect");
    EXPECT_NE(it, m->attrs.end());
    auto opt = (*it).second.try_cast<runtime::ObjectRef>();
    EXPECT_TRUE(opt.has_value());
    const auto* eff = opt.value().as<EffectNode>();
    EXPECT_NE(eff, nullptr);
    auto id_it = eff->attrs.find("id");
    EXPECT_NE(id_it, eff->attrs.end());
    return std::string((*id_it).second.cast<ffi::String>());
  };

  EXPECT_EQ(GetEffId(parent), "eff1");

  class ReplaceEff1 : public Mutator {
   public:
    ffi::Any visit_effect(const std::string& /*name*/, runtime::ObjectRef node) override {
      if (const auto* eff = node.as<EffectNode>()) {
        auto it = eff->attrs.find("id");
        if (it != eff->attrs.end()) {
          auto opt = (*it).second.try_cast<ffi::String>();
          if (opt.has_value() && std::string(opt.value()) == "eff1") {
            IOEffectModule eff2;
            eff2->attrs.Set("id", ffi::Any(ffi::String("eff2")));
            return ffi::Any(eff2);
          }
        }
      }
      return ffi::Any(node);
    }
  };

  ReplaceEff1 mutator;
  ffi::Any result = mutator.visit("", ffi::Any(parent));
  NNModule result_mod = result.cast<NNModule>();
  EXPECT_EQ(GetEffId(result_mod), "eff2");
}

// ---------------------------------------------------------------------------
// MutateParam
//
// Python equivalent:
//   class Parent: weight = Parameter((128, 64), "float16")
//   mutator replaces float16 param with float32
//   assert parent.weight.dtype == "float32"
// ---------------------------------------------------------------------------
TEST(NNVisitor, MutateParam) {
  ffi::Array<ffi::Any> shape{ffi::Any(int64_t(128)), ffi::Any(int64_t(64))};
  TensorNode* raw = TensorNode::MakePlaceholder(shape, "float16", "weight");
  NNParameter weight(raw->expr, std::nullopt, {});
  delete raw;

  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("weight", ffi::Any(weight));
  NNModule parent(attrs);

  EXPECT_EQ(std::string(weight->GetDtype()), "float16");

  class UpcastParam : public Mutator {
   public:
    ffi::Any visit_param(const std::string& /*name*/, NNParameter node) override {
      if (std::string(node->GetDtype()) == "float16") {
        ffi::Array<ffi::Any> s{ffi::Any(int64_t(128)), ffi::Any(int64_t(64))};
        TensorNode* raw = TensorNode::MakePlaceholder(s, "float32", "weight");
        NNParameter p(raw->expr, std::nullopt, {});
        delete raw;
        return ffi::Any(p);
      }
      return ffi::Any(node);
    }
  };

  UpcastParam mutator;
  mutator.visit("", ffi::Any(parent));

  // The parent's attrs["weight"] should now be float32.
  auto it = parent->attrs.find("weight");
  ASSERT_NE(it, parent->attrs.end());
  auto opt = (*it).second.try_cast<NNParameter>();
  ASSERT_TRUE(opt.has_value());
  EXPECT_EQ(std::string(opt.value()->GetDtype()), "float32");
}

// ---------------------------------------------------------------------------
// MutateRecursively
//
// Python equivalent:
//   class Sub: weight = Parameter((128,64), "float16")
//   class Parent: mod = Sub()
//   mutator upgrades float16 → float32
//   assert parent.mod.weight.dtype == "float32"
// ---------------------------------------------------------------------------
TEST(NNVisitor, MutateRecursively) {
  ffi::Array<ffi::Any> shape{ffi::Any(int64_t(128)), ffi::Any(int64_t(64))};
  TensorNode* raw = TensorNode::MakePlaceholder(shape, "float16", "weight");
  NNParameter weight(raw->expr, std::nullopt, {});
  delete raw;

  ffi::Map<ffi::String, ffi::Any> sub_attrs;
  sub_attrs.Set("weight", ffi::Any(weight));
  NNModule sub(sub_attrs);

  ffi::Map<ffi::String, ffi::Any> parent_attrs;
  parent_attrs.Set("mod", ffi::Any(sub));
  NNModule parent(parent_attrs);

  class UpcastParam : public Mutator {
   public:
    ffi::Any visit_param(const std::string& /*name*/, NNParameter node) override {
      if (std::string(node->GetDtype()) == "float16") {
        ffi::Array<ffi::Any> s{ffi::Any(int64_t(128)), ffi::Any(int64_t(64))};
        TensorNode* raw = TensorNode::MakePlaceholder(s, "float32", "weight");
        NNParameter p(raw->expr, std::nullopt, {});
        delete raw;
        return ffi::Any(p);
      }
      return ffi::Any(node);
    }
  };

  UpcastParam mutator;
  mutator.visit("", ffi::Any(parent));

  // Navigate: parent.attrs["mod"].attrs["weight"]
  auto mod_it = parent->attrs.find("mod");
  ASSERT_NE(mod_it, parent->attrs.end());
  auto opt_sub = (*mod_it).second.try_cast<NNModule>();
  ASSERT_TRUE(opt_sub.has_value());
  auto w_it = opt_sub.value()->attrs.find("weight");
  ASSERT_NE(w_it, opt_sub.value()->attrs.end());
  auto opt_w = (*w_it).second.try_cast<NNParameter>();
  ASSERT_TRUE(opt_w.has_value());
  EXPECT_EQ(std::string(opt_w.value()->GetDtype()), "float32");
}

// ---------------------------------------------------------------------------
// DefaultVisitParam
//
// The base Mutator::visit_param returns the node unchanged.
// ---------------------------------------------------------------------------
TEST(NNVisitor, DefaultVisitParam) {
  NNParameter p = MakeParam4("float32");
  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set("w", ffi::Any(p));
  NNModule parent(attrs);

  Mutator mutator;
  mutator.visit("", ffi::Any(parent));

  auto it = parent->attrs.find("w");
  ASSERT_NE(it, parent->attrs.end());
  EXPECT_EQ(ParamDtype((*it).second), "float32");
}

// ---------------------------------------------------------------------------
// DefaultVisitModule
//
// The base Mutator::visit_module returns the node unchanged.
// ---------------------------------------------------------------------------
TEST(NNVisitor, DefaultVisitModule) {
  ffi::Map<ffi::String, ffi::Any> sub_attrs;
  sub_attrs.Set("tag", ffi::Any(ffi::String("original")));
  NNModule sub(sub_attrs);

  ffi::Map<ffi::String, ffi::Any> parent_attrs;
  parent_attrs.Set("sub", ffi::Any(sub));
  NNModule parent(parent_attrs);

  Mutator mutator;
  ffi::Any result = mutator.visit("", ffi::Any(parent));
  NNModule result_mod = result.cast<NNModule>();

  auto it = result_mod->attrs.find("sub");
  ASSERT_NE(it, result_mod->attrs.end());
  auto opt = (*it).second.try_cast<NNModule>();
  ASSERT_TRUE(opt.has_value());
  auto tag_it = opt.value()->attrs.find("tag");
  ASSERT_NE(tag_it, opt.value()->attrs.end());
  EXPECT_EQ(std::string((*tag_it).second.cast<ffi::String>()), "original");
}

// ---------------------------------------------------------------------------
// NoopOnNonModule
//
// visit() on a plain integer Any returns it unchanged.
// ---------------------------------------------------------------------------
TEST(NNVisitor, NoopOnNonModule) {
  Mutator mutator;
  ffi::Any val = ffi::Any(int64_t(42));
  ffi::Any result = mutator.visit("x", val);
  EXPECT_EQ(result.cast<int64_t>(), 42);
}

// ---------------------------------------------------------------------------
// ModuleListNestedModuleDict
//
// Python equivalent:
//   ml = ModuleList([
//       ModuleDict({"a": Leaf("float32"), "b": Leaf("float16")}),
//       ModuleDict({"c": Leaf("float64")}),
//   ])
//   recorder.visit("root", ml)
//   assert "root.0.a.param" in seen_names
//   assert "root.0.b.param" in seen_names
//   assert "root.1.c.param" in seen_names
// ---------------------------------------------------------------------------
TEST(NNVisitor, ModuleListNestedModuleDict) {
  ffi::Map<ffi::String, ffi::Any> dict0_mods;
  dict0_mods.Set("a", ffi::Any(LeafModule("float32")));
  dict0_mods.Set("b", ffi::Any(LeafModule("float16")));
  ModuleDict dict0(dict0_mods);

  ffi::Map<ffi::String, ffi::Any> dict1_mods;
  dict1_mods.Set("c", ffi::Any(LeafModule("float64")));
  ModuleDict dict1(dict1_mods);

  ModuleList ml(ffi::Array<ffi::Any>{ffi::Any(dict0), ffi::Any(dict1)});

  RecorderMutator recorder;
  recorder.visit("root", ffi::Any(ml));

  std::map<std::string, std::string> seen_map;
  for (const auto& [name, dtype] : recorder.seen) seen_map[name] = dtype;

  EXPECT_NE(seen_map.find("root.0.a.param"), seen_map.end());
  EXPECT_NE(seen_map.find("root.0.b.param"), seen_map.end());
  EXPECT_NE(seen_map.find("root.1.c.param"), seen_map.end());
}

// ---------------------------------------------------------------------------
// ParamDtypeUpgrade
//
// Python equivalent:
//   class Root:
//       a = Leaf("float16")
//       b = Leaf("float32")
//       c = ModuleList([Leaf("float16"), Leaf("float64")])
//       d = ModuleDict({"x": Leaf("float16"), "y": Leaf("float32")})
//   mutator upgrades float16 → float32 everywhere
// ---------------------------------------------------------------------------
TEST(NNVisitor, ParamDtypeUpgrade) {
  // Build the tree.
  ModuleList c_list(ffi::Array<ffi::Any>{ffi::Any(LeafModule("float16")),
                                          ffi::Any(LeafModule("float64"))});

  ffi::Map<ffi::String, ffi::Any> d_mods;
  d_mods.Set("x", ffi::Any(LeafModule("float16")));
  d_mods.Set("y", ffi::Any(LeafModule("float32")));
  ModuleDict d_dict(d_mods);

  ffi::Map<ffi::String, ffi::Any> root_attrs;
  root_attrs.Set("a", ffi::Any(LeafModule("float16")));
  root_attrs.Set("b", ffi::Any(LeafModule("float32")));
  root_attrs.Set("c", ffi::Any(c_list));
  root_attrs.Set("d", ffi::Any(d_dict));
  NNModule root(root_attrs);

  // Upgrade mutator.
  class Upgrade : public Mutator {
   public:
    ffi::Any visit_param(const std::string& /*name*/, NNParameter node) override {
      if (std::string(node->GetDtype()) == "float16") {
        ffi::Array<ffi::Any> s{ffi::Any(int64_t(4))};
        TensorNode* raw = TensorNode::MakePlaceholder(s, "float32", "param");
        NNParameter p(raw->expr, std::nullopt, {});
        delete raw;
        return ffi::Any(p);
      }
      return ffi::Any(node);
    }
  };

  Upgrade mutator;
  mutator.visit("", ffi::Any(root));

  // Helper: get the dtype of the "param" child of a module stored in attrs.
  auto GetLeafDtype = [](ffi::Any val) -> std::string {
    auto opt = val.try_cast<NNModule>();
    EXPECT_TRUE(opt.has_value());
    auto it = opt.value()->attrs.find("param");
    EXPECT_NE(it, opt.value()->attrs.end());
    return ParamDtype((*it).second);
  };

  // root.a.param: float16 → float32
  EXPECT_EQ(GetLeafDtype(root->attrs.at("a")), "float32");
  // root.b.param: float32 unchanged
  EXPECT_EQ(GetLeafDtype(root->attrs.at("b")), "float32");

  // root.c[0].param: float16 → float32
  auto c_opt = root->attrs.at("c").try_cast<ModuleList>();
  ASSERT_TRUE(c_opt.has_value());
  EXPECT_EQ(GetLeafDtype(c_opt.value()->modules[0]), "float32");
  // root.c[1].param: float64 unchanged
  EXPECT_EQ(GetLeafDtype(c_opt.value()->modules[1]), "float64");

  // root.d["x"].param: float16 → float32
  auto d_opt = root->attrs.at("d").try_cast<ModuleDict>();
  ASSERT_TRUE(d_opt.has_value());
  EXPECT_EQ(GetLeafDtype(d_opt.value()->modules.at("x")), "float32");
  // root.d["y"].param: float32 unchanged
  EXPECT_EQ(GetLeafDtype(d_opt.value()->modules.at("y")), "float32");
}

}  // namespace testing
}  // namespace nn
}  // namespace frontend
}  // namespace relax
}  // namespace tvm
