/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_RV64_JIT_RVV_ELTWISE_EMITTER_HPP
#define CPU_RV64_JIT_RVV_ELTWISE_EMITTER_HPP

#include <cstdint>

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct eltwise_aux_regs_t {
    Xbyak_riscv::VReg tmp0;
    Xbyak_riscv::VReg tmp1;
    Xbyak_riscv::VReg tmp2;
    Xbyak_riscv::VReg tmp3;
    Xbyak_riscv::FReg alpha;
    Xbyak_riscv::FReg beta;
    Xbyak_riscv::FReg zero;
    Xbyak_riscv::FReg one;
    Xbyak_riscv::FReg ftmp0;
    Xbyak_riscv::FReg ftmp1;
    Xbyak_riscv::Reg tmp_reg0;
    Xbyak_riscv::Reg tmp_reg1;
};

struct jit_rvv_eltwise_fwd_emitter_t {
    explicit jit_rvv_eltwise_fwd_emitter_t(jit_generator_t *jit);

    void load_f32_bits(const Xbyak_riscv::FReg &dst,
            const Xbyak_riscv::Reg &tmp, uint32_t bits);
    void load_f32_const(const Xbyak_riscv::FReg &dst,
            const Xbyak_riscv::Reg &tmp, float value);
    void broadcast_f32_bits(const eltwise_aux_regs_t &r,
            const Xbyak_riscv::VReg &dst, uint32_t bits);
    void broadcast_f32_const(const eltwise_aux_regs_t &r,
            const Xbyak_riscv::VReg &dst, float value);
    void clamp_f32(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            float lower, float upper);
    void mul_f32_const(const eltwise_aux_regs_t &r,
            const Xbyak_riscv::VReg &dst, const Xbyak_riscv::VReg &src,
            float value);
    void horner_f32(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &x, const float *coeffs, int coeff_count);
    void horner_f32_bits(const eltwise_aux_regs_t &r,
            const Xbyak_riscv::VReg &dst, const Xbyak_riscv::VReg &x,
            const uint32_t *coeffs, int coeff_count);

    void abs(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void clip(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void hardsigmoid(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void hardswish(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void linear(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void relu(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void sqrt(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void square(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void leakyrelu(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void round(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);

    void exp(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void erf_exp_complement(const eltwise_aux_regs_t &r,
            const Xbyak_riscv::VReg &dst, const Xbyak_riscv::VReg &src);
    void tanh(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void erf(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void reciprocal(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void sigmoid(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void swish(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);

    void elu(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void gelu_tanh(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);
    void gelu_erf(const eltwise_aux_regs_t &r, const Xbyak_riscv::VReg &dst,
            const Xbyak_riscv::VReg &src);

private:
    jit_generator_t *jit_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_ELTWISE_EMITTER_HPP
