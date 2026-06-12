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

#include <assert.h>
#include <cstdint>
#include <cstring>

#include "cpu/rv64/jit_rvv_eltwise_emitter.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

constexpr struct {
    float LowerRange;
    float UpperRange;
    uint32_t RoundingBias;
    uint32_t Log2Reciprocal;
    uint32_t Log2High;
    uint32_t Log2Low;
    uint32_t poly_0;
    uint32_t poly_1;
    uint32_t poly_2;
    uint32_t poly_3;
    uint32_t poly_4;
    uint32_t poly_56;
    int32_t MinimumExponent;
    int32_t MaximumExponent;
} exp_constants = {
        -103.9720840454f,
        88.7762626647950f,
        0x4b400000,
        0x3fb8aa3b,
        0xbf317200,
        0xb5bfbe8e,
        0x3ab4a000,
        0x3c092f6e,
        0x3d2aadad,
        0x3e2aaa28,
        0x3efffffb,
        0x3f800000,
        int32_t(0xc1000000),
        int32_t(0x3f800000),
};

constexpr struct {
    float LowerRange;
    float UpperRange;
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
} tanh_constants = {
        -9.0f,
        9.0f,
        -2.76076847742355e-16f,
        2.00018790482477e-13f,
        -8.60467152213735e-11f,
        5.12229709037114e-08f,
        1.48572235717979e-05f,
        6.37261928875436e-04f,
        4.89352455891786e-03f,
        1.19825839466702e-06f,
        1.18534705686654e-04f,
        2.26843463243900e-03f,
        4.89352518554385e-03f,
};

constexpr struct {
    float alpha_13;
    float alpha_11;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float GELU_COEF_A;
    float GELU_QUICK_COEF;
    float SQRT_2_OVER_PI;
} gelu_tanh_constants = {
        -2.76076847742355e-16f,
        2.00018790482477e-13f,
        -8.60467152213735e-11f,
        5.12229709037114e-08f,
        1.48572235717979e-05f,
        6.37261928875436e-04f,
        4.89352455891786e-03f,
        1.19825839466702e-06f,
        1.18534705686654e-04f,
        2.26843463243900e-03f,
        4.89352518554385e-03f,
        0.044715f,
        -1.702f,
        0.79788456080286535587989211986876f,
};

constexpr struct {
    float ErfUpperAbsRange;
    float ErfSplitBoundary;
    float ErfSMALL_P0;
    float ErfSMALL_P1;
    float ErfSMALL_P2;
    float ErfSMALL_P3;
    float ErfSMALL_P4;
    float ErfSMALL_P5_Minus_One;
    float ErfReserved0;
    float ErfBIG_P0;
    float ErfBIG_P1;
    float ErfBIG_P2;
    float ErfBIG_P3;
    float ErfBIG_P4;
    float ErfBIG_P5;
    float ErfBIG_P6_Minus_One;
    float ErfNegZero;
    float ErfOne;
    float Exp_UpperRange;
    float Exp_LowerRange;
    float Exp_Log2Reciprocal;
    float Exp_log2_hi;
    float Exp_log2_lo;
    float Exp_P0;
    float Exp_P1;
    float Exp_P2;
    float Exp_P3;
    float Exp_P4;
    float Exp_P5;
    float Exp_P6;
    float Exp_C;
    int32_t Exp_X7F;
} erf_constants = {
        3.925f,
        0.921875f,
        -5.99104969e-4f,
        4.99339588e-3f,
        -2.67667342e-2f,
        1.12818025e-1f,
        -3.76124859e-1f,
        1.28379151e-1f,
        0.0f,
        1.72948930e-5f,
        -3.83208680e-4f,
        3.88393435e-3f,
        -2.42545605e-2f,
        1.06777847e-1f,
        6.34846687e-1f,
        1.28717512e-1f,
        -0.0f,
        1.0f,
        // Independent parameters to calculate Exp for Erff()
        88.3762626647950f,
        -88.3762626647949f,
        1.44269504088896341f,
        -6.93145752e-1f,
        -1.42860677e-6f,
        1.38319808e-3f,
        8.37550033e-3f,
        4.16689515e-2f,
        1.66664466e-1f,
        4.99999851e-1f,
        1.00000000e+0f,
        1.00000000e+0f,
        1.25829120e+7f,
        127,
};

constexpr uint32_t exp_poly[] = {exp_constants.poly_0, exp_constants.poly_1,
        exp_constants.poly_2, exp_constants.poly_3, exp_constants.poly_4,
        exp_constants.poly_56};
constexpr float tanh_num_poly[] = {tanh_constants.alpha_13,
        tanh_constants.alpha_11, tanh_constants.alpha_9, tanh_constants.alpha_7,
        tanh_constants.alpha_5, tanh_constants.alpha_3, tanh_constants.alpha_1};
constexpr float tanh_den_poly[] = {tanh_constants.beta_6, tanh_constants.beta_4,
        tanh_constants.beta_2, tanh_constants.beta_0};
constexpr float erf_small_poly[]
        = {erf_constants.ErfSMALL_P0, erf_constants.ErfSMALL_P1,
                erf_constants.ErfSMALL_P2, erf_constants.ErfSMALL_P3,
                erf_constants.ErfSMALL_P4, erf_constants.ErfSMALL_P5_Minus_One};
constexpr float erf_big_poly[] = {erf_constants.ErfBIG_P0,
        erf_constants.ErfBIG_P1, erf_constants.ErfBIG_P2,
        erf_constants.ErfBIG_P3, erf_constants.ErfBIG_P4,
        erf_constants.ErfBIG_P5, erf_constants.ErfBIG_P6_Minus_One};
constexpr float erf_exp_poly[] = {erf_constants.Exp_P0, erf_constants.Exp_P1,
        erf_constants.Exp_P2, erf_constants.Exp_P3, erf_constants.Exp_P4,
        erf_constants.Exp_P5, erf_constants.Exp_P6};

template <typename T, int N>
constexpr int array_size(const T (&)[N]) {
    return N;
}

uint32_t float_to_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

jit_rvv_eltwise_fwd_emitter_t::jit_rvv_eltwise_fwd_emitter_t(
        jit_generator_t *jit)
    : jit_(jit) {}

void jit_rvv_eltwise_fwd_emitter_t::load_f32_bits(
        const FReg &dst, const Reg &tmp, uint32_t bits) {
    jit_->li(tmp, static_cast<int64_t>(bits));
    jit_->fmv_w_x(dst, tmp);
}

void jit_rvv_eltwise_fwd_emitter_t::load_f32_const(
        const FReg &dst, const Reg &tmp, float value) {
    load_f32_bits(dst, tmp, float_to_bits(value));
}

void jit_rvv_eltwise_fwd_emitter_t::broadcast_f32_bits(
        const eltwise_aux_regs_t &r, const VReg &dst, uint32_t bits) {
    load_f32_bits(r.ftmp0, r.tmp_reg0, bits);
    jit_->vfmv_v_f(dst, r.ftmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::broadcast_f32_const(
        const eltwise_aux_regs_t &r, const VReg &dst, float value) {
    broadcast_f32_bits(r, dst, float_to_bits(value));
}

void jit_rvv_eltwise_fwd_emitter_t::clamp_f32(const eltwise_aux_regs_t &r,
        const VReg &dst, float lower, float upper) {
    load_f32_const(r.ftmp0, r.tmp_reg0, lower);
    jit_->vfmax_vf(dst, dst, r.ftmp0);
    load_f32_const(r.ftmp0, r.tmp_reg0, upper);
    jit_->vfmin_vf(dst, dst, r.ftmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::mul_f32_const(const eltwise_aux_regs_t &r,
        const VReg &dst, const VReg &src, float value) {
    load_f32_const(r.ftmp0, r.tmp_reg0, value);
    jit_->vfmul_vf(dst, src, r.ftmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::horner_f32(const eltwise_aux_regs_t &r,
        const VReg &dst, const VReg &x, const float *coeffs, int coeff_count) {
    assert(coeff_count > 0);
    broadcast_f32_const(r, dst, coeffs[0]);
    for (int i = 1; i < coeff_count; ++i) {
        jit_->vfmul_vv(dst, dst, x);
        load_f32_const(r.ftmp0, r.tmp_reg0, coeffs[i]);
        jit_->vfadd_vf(dst, dst, r.ftmp0);
    }
}

void jit_rvv_eltwise_fwd_emitter_t::horner_f32_bits(const eltwise_aux_regs_t &r,
        const VReg &dst, const VReg &x, const uint32_t *coeffs,
        int coeff_count) {
    assert(coeff_count > 0);
    broadcast_f32_bits(r, dst, coeffs[0]);
    for (int i = 1; i < coeff_count; ++i) {
        jit_->vfmul_vv(dst, dst, x);
        load_f32_bits(r.ftmp0, r.tmp_reg0, coeffs[i]);
        jit_->vfadd_vf(dst, dst, r.ftmp0);
    }
}

void jit_rvv_eltwise_fwd_emitter_t::abs(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    UNUSED(r);
    jit_->vfabs_v(dst, src);
}

void jit_rvv_eltwise_fwd_emitter_t::clip(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmax_vf(dst, src, r.alpha);
    jit_->vfmin_vf(dst, dst, r.beta);
}

void jit_rvv_eltwise_fwd_emitter_t::hardsigmoid(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmul_vf(dst, src, r.alpha);
    jit_->vfadd_vf(dst, dst, r.beta);
    jit_->vfmax_vf(dst, dst, r.zero);
    jit_->vfmin_vf(dst, dst, r.one);
}

void jit_rvv_eltwise_fwd_emitter_t::hardswish(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmul_vf(r.tmp0, src, r.alpha);
    jit_->vfadd_vf(r.tmp0, r.tmp0, r.beta);
    jit_->vfmax_vf(r.tmp0, r.tmp0, r.zero);
    jit_->vfmin_vf(r.tmp0, r.tmp0, r.one);
    jit_->vfmul_vv(dst, src, r.tmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::linear(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmul_vf(dst, src, r.alpha);
    jit_->vfadd_vf(dst, dst, r.beta);
}

void jit_rvv_eltwise_fwd_emitter_t::relu(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    const VReg v_mask(0);
    jit_->vmfgt_vf(v_mask, src, r.zero);
    jit_->vfmul_vf(r.tmp0, src, r.alpha);
    jit_->vmerge_vvm(dst, r.tmp0, src);
}

void jit_rvv_eltwise_fwd_emitter_t::sqrt(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    UNUSED(r);
    jit_->vfsqrt_v(dst, src);
}

void jit_rvv_eltwise_fwd_emitter_t::square(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    UNUSED(r);
    jit_->vfmul_vv(dst, src, src);
}

void jit_rvv_eltwise_fwd_emitter_t::leakyrelu(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmax_vf(dst, src, r.zero);
    jit_->vfmin_vf(r.tmp0, src, r.zero);
    jit_->vfmul_vf(r.tmp0, r.tmp0, r.alpha);
    jit_->vfadd_vv(dst, dst, r.tmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::round(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfcvt_x_f_v(r.tmp0, src);
    jit_->vfcvt_f_x_v(dst, r.tmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::exp(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vmv_v_v(dst, src);
    clamp_f32(r, dst, exp_constants.LowerRange, exp_constants.UpperRange);

    broadcast_f32_bits(r, r.tmp0, exp_constants.RoundingBias);
    load_f32_bits(r.ftmp0, r.tmp_reg0, exp_constants.Log2Reciprocal);
    jit_->vfmacc_vf(r.tmp0, r.ftmp0, dst);

    load_f32_bits(r.ftmp0, r.tmp_reg0, exp_constants.RoundingBias);
    jit_->vfsub_vf(r.tmp1, r.tmp0, r.ftmp0);
    load_f32_bits(r.ftmp0, r.tmp_reg0, exp_constants.Log2High);
    jit_->vfmacc_vf(dst, r.ftmp0, r.tmp1);
    load_f32_bits(r.ftmp0, r.tmp_reg0, exp_constants.Log2Low);
    jit_->vfmacc_vf(dst, r.ftmp0, r.tmp1);

    horner_f32_bits(r, r.tmp2, dst, exp_poly, array_size(exp_poly));

    jit_->vsll_vi(r.tmp0, r.tmp0, 23);
    jit_->li(r.tmp_reg0, exp_constants.MaximumExponent);
    jit_->li(r.tmp_reg1, exp_constants.MinimumExponent);
    jit_->vmin_vx(r.tmp1, r.tmp0, r.tmp_reg0);
    jit_->vmax_vx(r.tmp1, r.tmp1, r.tmp_reg1);
    jit_->vsub_vv(r.tmp0, r.tmp0, r.tmp1);
    jit_->vadd_vx(r.tmp0, r.tmp0, r.tmp_reg0);
    jit_->vadd_vx(r.tmp1, r.tmp1, r.tmp_reg0);

    jit_->vfmul_vv(dst, dst, r.tmp0);
    jit_->vfmadd_vv(r.tmp2, dst, r.tmp0);
    jit_->vfmul_vv(dst, r.tmp2, r.tmp1);
}

void jit_rvv_eltwise_fwd_emitter_t::erf_exp_complement(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vmv_v_v(r.tmp1, src);
    load_f32_const(r.ftmp0, r.tmp_reg0, erf_constants.Exp_LowerRange);
    jit_->vfmax_vf(r.tmp1, r.tmp1, r.ftmp0);

    broadcast_f32_const(r, r.tmp0, erf_constants.Exp_Log2Reciprocal);
    jit_->vfmul_vv(r.tmp0, r.tmp0, r.tmp1);
    load_f32_const(r.ftmp0, r.tmp_reg0, erf_constants.Exp_C);
    jit_->vfadd_vf(r.tmp0, r.tmp0, r.ftmp0);
    jit_->vfsub_vf(r.tmp0, r.tmp0, r.ftmp0);

    load_f32_const(r.ftmp0, r.tmp_reg0, erf_constants.Exp_log2_hi);
    jit_->vfmacc_vf(r.tmp1, r.ftmp0, r.tmp0);
    load_f32_const(r.ftmp0, r.tmp_reg0, erf_constants.Exp_log2_lo);
    jit_->vfmacc_vf(r.tmp1, r.ftmp0, r.tmp0);

    horner_f32(r, dst, r.tmp1, erf_exp_poly, array_size(erf_exp_poly));

    jit_->vfcvt_x_f_v(r.tmp0, r.tmp0);
    jit_->li(r.tmp_reg0, erf_constants.Exp_X7F);
    jit_->vadd_vx(r.tmp0, r.tmp0, r.tmp_reg0);
    jit_->vsll_vi(r.tmp0, r.tmp0, 23);
    jit_->vfmul_vv(dst, dst, r.tmp0);
    jit_->vfrsub_vf(dst, dst, r.one);
}

void jit_rvv_eltwise_fwd_emitter_t::tanh(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vmv_v_v(dst, src);
    clamp_f32(r, dst, tanh_constants.LowerRange, tanh_constants.UpperRange);

    jit_->vfmul_vv(r.tmp0, dst, dst);
    horner_f32(r, r.tmp1, r.tmp0, tanh_num_poly, array_size(tanh_num_poly));
    jit_->vfmul_vv(r.tmp1, r.tmp1, dst);
    horner_f32(r, r.tmp2, r.tmp0, tanh_den_poly, array_size(tanh_den_poly));

    jit_->vfdiv_vv(dst, r.tmp1, r.tmp2);
}

void jit_rvv_eltwise_fwd_emitter_t::erf(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    const VReg v_mask(0);

    broadcast_f32_const(r, r.tmp3, -0.0f);
    jit_->vand_vv(r.tmp2, src, r.tmp3);
    jit_->vfabs_v(r.tmp0, src);
    jit_->vfmul_vv(r.tmp1, r.tmp0, r.tmp0);

    horner_f32(r, dst, r.tmp1, erf_small_poly, array_size(erf_small_poly));
    jit_->vfmul_vv(dst, dst, r.tmp0);
    jit_->vfadd_vv(dst, dst, r.tmp0);

    load_f32_const(r.ftmp0, r.tmp_reg0, erf_constants.ErfSplitBoundary);
    jit_->vmfgt_vf(v_mask, r.tmp0, r.ftmp0);
    load_f32_const(r.ftmp0, r.tmp_reg0, erf_constants.ErfUpperAbsRange);
    jit_->vfmin_vf(r.tmp0, r.tmp0, r.ftmp0);

    horner_f32(r, r.tmp1, r.tmp0, erf_big_poly, array_size(erf_big_poly));
    jit_->vfmul_vv(r.tmp1, r.tmp1, r.tmp0);
    jit_->vfadd_vv(r.tmp1, r.tmp1, r.tmp0);

    jit_->vfneg_v(r.tmp1, r.tmp1);
    erf_exp_complement(r, r.tmp3, r.tmp1);

    jit_->vmerge_vvm(r.tmp3, dst, r.tmp3);
    jit_->vor_vv(dst, r.tmp3, r.tmp2);
}

void jit_rvv_eltwise_fwd_emitter_t::reciprocal(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfrec7_v(dst, src);
    for (int i = 0; i < 2; ++i) {
        broadcast_f32_const(r, r.tmp2, 2.0f);
        jit_->vfnmsac_vv(r.tmp2, src, dst);
        jit_->vfmul_vv(dst, dst, r.tmp2);
    }
}

void jit_rvv_eltwise_fwd_emitter_t::sigmoid(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    const VReg v_mask(0);

    jit_->vfabs_v(r.tmp3, src);
    jit_->vmfeq_vv(v_mask, src, r.tmp3);
    jit_->vfneg_v(r.tmp3, r.tmp3);
    exp(r, r.tmp3, r.tmp3);

    jit_->vfadd_vf(r.tmp0, r.tmp3, r.one);
    reciprocal(r, r.tmp1, r.tmp0);
    jit_->vfmul_vv(r.tmp3, r.tmp3, r.tmp1);
    jit_->vmerge_vvm(dst, r.tmp3, r.tmp1);
}
void jit_rvv_eltwise_fwd_emitter_t::swish(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmul_vf(r.tmp0, src, r.alpha);
    sigmoid(r, r.tmp0, r.tmp0);
    jit_->vfmul_vv(dst, src, r.tmp0);
}

void jit_rvv_eltwise_fwd_emitter_t::elu(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    const VReg v_mask(0);
    exp(r, r.tmp3, src);
    jit_->vfsub_vf(r.tmp3, r.tmp3, r.one);
    jit_->vfmul_vf(r.tmp3, r.tmp3, r.alpha);
    jit_->vmflt_vf(v_mask, src, r.zero);
    jit_->vmerge_vvm(dst, src, r.tmp3);
}

void jit_rvv_eltwise_fwd_emitter_t::gelu_tanh(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    jit_->vfmul_vv(r.tmp0, src, src);
    mul_f32_const(r, r.tmp0, r.tmp0, gelu_tanh_constants.GELU_COEF_A);
    jit_->vfadd_vf(r.tmp0, r.tmp0, r.one);
    jit_->vfmul_vv(r.tmp0, r.tmp0, src);
    mul_f32_const(r, r.tmp0, r.tmp0, gelu_tanh_constants.SQRT_2_OVER_PI);

    tanh(r, r.tmp3, r.tmp0);
    jit_->vfadd_vf(r.tmp3, r.tmp3, r.one);
    jit_->vfmul_vv(dst, r.tmp3, src);
    mul_f32_const(r, dst, dst, 0.5f);
    clamp_f32(r, dst, -256.0f, 256.0f);
}

void jit_rvv_eltwise_fwd_emitter_t::gelu_erf(
        const eltwise_aux_regs_t &r, const VReg &dst, const VReg &src) {
    mul_f32_const(r, r.tmp0, src, 0.70710678118654752440f);
    erf(r, dst, r.tmp0);

    jit_->vfadd_vf(dst, dst, r.one);
    mul_f32_const(r, r.tmp0, src, 0.5f);
    jit_->vfmul_vv(dst, r.tmp0, dst);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
