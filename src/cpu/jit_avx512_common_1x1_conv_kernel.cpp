/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include <float.h>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_common_1x1_conv_kernel.hpp"

#include "cpu_barrier.hpp"
#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;
namespace barrier = simple_barrier;

namespace {

float loss_ratio(int amount, int divider)
{
    return float(rnd_up(amount, divider) - amount) / rnd_up(amount, divider);
}

int best_divider(int value, int min_divider, int max_divider,
                        bool find_max, int step = 1)
{
    max_divider = nstl::max(1, nstl::min(max_divider, value));
    min_divider = nstl::max(1, nstl::min(min_divider, max_divider));

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

}

void jit_avx512_common_1x1_conv_kernel::bcast_loop(int load_loop_blk)
{
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(reg_output_data, EVEX_compress_addr(rsp, output_data));
    mov(aux_reg_output_data, reg_output_data);

    if (one_of(jcp.prop_kind, backward_data)){
        vmovq(EVEX_compress_addr(rsp, aux_relu_src), xtmp);
        vmovq(EVEX_compress_addr(rsp, aux_bn_src), xtmp2);
    }

    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_offt));

    if (jcp.ver == ver_4fma)
    {
        Label bcast_loop;
        Label bcast_loop_wraparound;
        Label bcast_loop_out;
        Label bcast_loop_ur_full;

        cmp(bcast_loop_iter, jcp.ur);
        jle(bcast_loop_wraparound, T_NEAR);

        L(bcast_loop); {
            assert(jcp.bcast_block % jcp.ur == 0);
            int num_substeps = jcp.bcast_block / jcp.ur;
            assert(num_substeps > 0 && num_substeps < 10);
            for (int i = 0; i < num_substeps; i++) {
                reduce_loop(load_loop_blk, jcp.ur, i, false);
                if (i < num_substeps - 1) {
                    add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                    add(aux_reg_output_data, jcp.bcast_loop_output_substep);
                }
                else {
                    add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                        - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                    add(aux_reg_output_data, jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep);
                }
            }
            sub(bcast_loop_iter, jcp.bcast_block);
            cmp(bcast_loop_iter, jcp.bcast_block);
            jg(bcast_loop, T_NEAR);
        }

        L(bcast_loop_wraparound);
        if (jcp.ur_tail) {
            je(bcast_loop_ur_full, T_NEAR);
            reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
            jmp(bcast_loop_out, T_NEAR);
        }
        L(bcast_loop_ur_full);
        reduce_loop(load_loop_blk, jcp.ur, 0, true);
        L(bcast_loop_out);
    }
    else
    {
        Label bcast_loop;
        Label bcast_loop_tail;

        cmp(bcast_loop_iter, jcp.ur);
        jl(bcast_loop_tail, T_NEAR);

        L(bcast_loop); {
            assert(jcp.bcast_block % jcp.ur == 0);
            int num_substeps = jcp.bcast_block / jcp.ur;
            assert(num_substeps > 0 && num_substeps < 10);
            // pre-processing the coefficients for the BatchNorm layer on the forward_training.
            if (one_of(jcp.prop_kind, forward_training)) {
                mov(reg_last_flag, EVEX_compress_addr(rsp, last_flag));

                Label no_OC_FIRST;
                mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                cmp(reg_coff2, 0);
                jne(no_OC_FIRST);

                mov(reg_prev_mean, EVEX_compress_addr(rsp, prev_mean));
                mov(reg_prev_var, EVEX_compress_addr(rsp, prev_var));
                mov(reg_prev_src, EVEX_compress_addr(rsp, prev_src));

                vpxord(zmm_zero, zmm_zero, zmm_zero);
                uni_vmovups(zmean, zword[reg_prev_mean]);

                uni_vmovups(zsqrtvar, zword[reg_prev_var]);
                uni_vaddps(zsqrtvar, zsqrtvar, veps);
                vsqrtps(zsqrtvar, zsqrtvar);
                vdivps(zsqrtvar, vone, zsqrtvar);

                L(no_OC_FIRST);
            }
            // pre-processing the coefficients for the BatchNorm layer on the backward_training.
            if ((jcp.prop_kind == backward_data) &&
                (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
                mov(reg_last_flag, EVEX_compress_addr(rsp, last_flag));
                Label no_OC_FIRST;
                mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                cmp(reg_coff2, 0);
                jne(no_OC_FIRST, T_NEAR);

                mov(reg_next_mean, EVEX_compress_addr(rsp, next_mean));
                uni_vmovups(zmean, zword[reg_next_mean]);
                mov(reg_next_var, EVEX_compress_addr(rsp, next_var));
                uni_vmovups(zsqrtvar, zword[reg_next_var]);

                uni_vbroadcastss(bwd_eps, EVEX_compress_addr(rsp, eps));
                vaddps(zsqrtvar, zsqrtvar, bwd_eps);
                vsqrtps(zsqrtvar, zsqrtvar);
                uni_vbroadcastss(bwd_one, EVEX_compress_addr(rsp, one));
                vdivps(zsqrtvar, bwd_one, zsqrtvar);

                mov(reg_next_scale_shift, EVEX_compress_addr(rsp, next_scale_shift));
                vmovups(zgamma, zword[reg_next_scale_shift]);
                mov(reg_diff_gamma, EVEX_compress_addr(rsp, diff_gamma));
                vmovups(zdiff_gamma, zword[reg_diff_gamma]);
                mov(reg_diff_beta, EVEX_compress_addr(rsp, diff_beta));
                vmovups(zdiff_beta, zword[reg_diff_beta]);
                vmulps(zdiff_gamma, zdiff_gamma, zsqrtvar);

                uni_vbroadcastss(bwd_chan, EVEX_compress_addr(rsp, bwd_chan_size));
                vdivps(zdiff_beta, zdiff_beta, bwd_chan);
                vdivps(zdiff_gamma, zdiff_gamma, bwd_chan);

                L(no_OC_FIRST);
            }

            for (int i = 0; i < num_substeps; i++) {
                // Calculating the mean/variance for BatchNorm is only done on the first reduce_loop function (reduce_loop_OC_FIRST) call.
                if (one_of(jcp.prop_kind, forward_training)) {
                    Label no_OC_FIRST;
                    Label reduce_loop_done;
                    mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                    cmp(reg_coff2, 0);
                    jne(no_OC_FIRST, T_NEAR);
                    reduce_loop_OC_FIRST(load_loop_blk, jcp.ur, i, false);
                    jmp(reduce_loop_done, T_NEAR);

                    L(no_OC_FIRST);
                    reduce_loop(load_loop_blk, jcp.ur, i, false);
                    L(reduce_loop_done);
                }
                else if ((jcp.prop_kind == backward_data) &&
                        (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
                    Label no_OC_FIRST;
                    Label reduce_loop_done;
                    mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                    cmp(reg_coff2, 0);
                    jne(no_OC_FIRST, T_NEAR);
                    reduce_loop_OC_FIRST(load_loop_blk, jcp.ur, i, false);
                    jmp(reduce_loop_done, T_NEAR);

                    L(no_OC_FIRST);
                    reduce_loop(load_loop_blk, jcp.ur, i, false);
                    L(reduce_loop_done);
                }
                else
                    reduce_loop(load_loop_blk, jcp.ur, i, false);

                if (i < num_substeps - 1) {
                    if (one_of(jcp.prop_kind, forward_training)) {
                        Label no_OC_FIRST;
                        mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                        cmp(reg_coff2, 0);
                        jne(no_OC_FIRST);

                        mov(reg_prev_mean, EVEX_compress_addr(rsp, prev_mean));
                        mov(reg_prev_var, EVEX_compress_addr(rsp, prev_var));
                        mov(reg_prev_src, EVEX_compress_addr(rsp, prev_src));

                        uni_vmovups(zmean, zword[reg_prev_mean]);

                        uni_vmovups(zsqrtvar, zword[reg_prev_var]);
                        uni_vaddps(zsqrtvar, zsqrtvar, veps);
                        vsqrtps(zsqrtvar, zsqrtvar);
                        vdivps(zsqrtvar, vone, zsqrtvar);


                        L(no_OC_FIRST);
                    }

                    add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                    add(aux_reg_output_data, jcp.bcast_loop_output_substep);
                    if (one_of(jcp.prop_kind, backward_data)) {
                        mov(aux_reg_relu_src, EVEX_compress_addr(rsp, aux_relu_src));
                        add(aux_reg_relu_src, jcp.bcast_loop_output_substep);
                        mov(EVEX_compress_addr(rsp, aux_relu_src), aux_reg_relu_src);
                        mov(aux_reg_bn_src, EVEX_compress_addr(rsp, aux_bn_src));
                        add(aux_reg_bn_src, jcp.bcast_loop_output_substep);
                        mov(EVEX_compress_addr(rsp, aux_bn_src), aux_reg_bn_src);
                    }
                }
                else {
                    if (one_of(jcp.prop_kind, forward_training)) {
                        Label no_OC_FIRST;
                        mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                        cmp(reg_coff2, 0);
                        jne(no_OC_FIRST);

                        mov(reg_prev_mean, EVEX_compress_addr(rsp, prev_mean));
                        mov(reg_prev_var, EVEX_compress_addr(rsp, prev_var));

                        uni_vmovups(zmean, zword[reg_prev_mean]);

                        uni_vmovups(zsqrtvar, zword[reg_prev_var]);
                        uni_vaddps(zsqrtvar, zsqrtvar, veps);
                        vsqrtps(zsqrtvar, zsqrtvar);
                        vdivps(zsqrtvar, vone, zsqrtvar);

                        mov(reg_prev_src, EVEX_compress_addr(rsp, prev_src));
                        add(reg_prev_src, jcp.bcast_loop_bcast_step
                                    - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                        mov(EVEX_compress_addr(rsp, prev_src), reg_prev_src);
                        L(no_OC_FIRST);
                    }

                    if ((jcp.prop_kind == backward_data) &&
                        (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
                        mov(reg_last_flag, EVEX_compress_addr(rsp, last_flag));
                        Label no_OC_FIRST;
                        mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                        cmp(reg_coff2, 0);
                        jne(no_OC_FIRST, T_NEAR);

                        mov(reg_next_mean, EVEX_compress_addr(rsp, next_mean));
                        uni_vmovups(zmean, zword[reg_next_mean]);
                        mov(reg_next_var, EVEX_compress_addr(rsp, next_var));
                        uni_vmovups(zsqrtvar, zword[reg_next_var]);

                        uni_vbroadcastss(bwd_eps, EVEX_compress_addr(rsp, eps));
                        vaddps(zsqrtvar, zsqrtvar, bwd_eps);
                        vsqrtps(zsqrtvar, zsqrtvar);
                        uni_vbroadcastss(bwd_one, EVEX_compress_addr(rsp, one));
                        vdivps(zsqrtvar, bwd_one, zsqrtvar);

                        mov(reg_next_scale_shift, EVEX_compress_addr(rsp, next_scale_shift));
                        vmovups(zgamma, zword[reg_next_scale_shift]);
                        mov(reg_diff_gamma, EVEX_compress_addr(rsp, diff_gamma));
                        vmovups(zdiff_gamma, zword[reg_diff_gamma]);
                        mov(reg_diff_beta, EVEX_compress_addr(rsp, diff_beta));
                        vmovups(zdiff_beta, zword[reg_diff_beta]);
                        vmulps(zdiff_gamma, zdiff_gamma, zsqrtvar);

                        uni_vbroadcastss(bwd_chan, EVEX_compress_addr(rsp, bwd_chan_size));
                        vdivps(zdiff_beta, zdiff_beta, bwd_chan);
                        vdivps(zdiff_gamma, zdiff_gamma, bwd_chan);

                        mov(reg_conv_dst, EVEX_compress_addr(rsp, conv_dst));
                        add(reg_conv_dst, jcp.bcast_loop_bcast_step
                            - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                        mov(EVEX_compress_addr(rsp, conv_dst), reg_conv_dst);
                        mov(reg_diff_src, EVEX_compress_addr(rsp, diff_src));
                        add(reg_diff_src, jcp.bcast_loop_bcast_step
                            - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                        mov(EVEX_compress_addr(rsp, diff_src), reg_diff_src);

                        L(no_OC_FIRST);
                    }

                    add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                            - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                    add(aux_reg_output_data, jcp.bcast_loop_output_step
                            - (num_substeps - 1) * jcp.bcast_loop_output_substep);
                    if (one_of(jcp.prop_kind, backward_data)) {
                        mov(aux_reg_relu_src, EVEX_compress_addr(rsp, aux_relu_src));
                        add(aux_reg_relu_src, jcp.bcast_loop_output_step
                            - (num_substeps - 1) * jcp.bcast_loop_output_substep);
                        mov(EVEX_compress_addr(rsp, aux_relu_src), aux_reg_relu_src);
                        mov(aux_reg_bn_src, EVEX_compress_addr(rsp, aux_bn_src));
                        add(aux_reg_bn_src, jcp.bcast_loop_output_step
                                - (num_substeps - 1) * jcp.bcast_loop_output_substep);
                        mov(EVEX_compress_addr(rsp, aux_bn_src), aux_reg_bn_src);
                    }
                }
            }
            sub(bcast_loop_iter, jcp.bcast_block);
            cmp(bcast_loop_iter, jcp.bcast_block);
            jge(bcast_loop, T_NEAR);
        }

        L(bcast_loop_tail);
        if (jcp.ur_tail) {
            Label bcast_loop_tail_out;
            cmp(bcast_loop_iter, 0);
            jz(bcast_loop_tail_out, T_NEAR);
            if (one_of(jcp.prop_kind, forward_training)) {
                Label no_OC_FIRST;
                mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                cmp(reg_coff2, 0);
                jne(no_OC_FIRST);
                mov(reg_prev_mean, EVEX_compress_addr(rsp, prev_mean));
                mov(reg_prev_var, EVEX_compress_addr(rsp, prev_var));
                uni_vmovups(zmean, zword[reg_prev_mean]);
                uni_vmovups(zsqrtvar, zword[reg_prev_var]);
                uni_vaddps(zsqrtvar, zsqrtvar, veps);
                vsqrtps(zsqrtvar, zsqrtvar);
                vdivps(zsqrtvar, vone, zsqrtvar);

                L(no_OC_FIRST);
            }
            if ((jcp.prop_kind == backward_data) &&
                (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
                mov(reg_last_flag, EVEX_compress_addr(rsp, last_flag));
                Label no_OC_FIRST;
                mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                cmp(reg_coff2, 0);
                jne(no_OC_FIRST, T_NEAR);

                mov(reg_next_mean, EVEX_compress_addr(rsp, next_mean));
                uni_vmovups(zmean, zword[reg_next_mean]);
                mov(reg_next_var, EVEX_compress_addr(rsp, next_var));
                uni_vmovups(zsqrtvar, zword[reg_next_var]);

                uni_vbroadcastss(bwd_eps, EVEX_compress_addr(rsp, eps));
                vaddps(zsqrtvar, zsqrtvar, bwd_eps);
                vsqrtps(zsqrtvar, zsqrtvar);
                uni_vbroadcastss(bwd_one, EVEX_compress_addr(rsp, one));
                vdivps(zsqrtvar, bwd_one, zsqrtvar);

                mov(reg_next_scale_shift, EVEX_compress_addr(rsp, next_scale_shift));
                vmovups(zgamma, zword[reg_next_scale_shift]);
                mov(reg_diff_gamma, EVEX_compress_addr(rsp, diff_gamma));
                vmovups(zdiff_gamma, zword[reg_diff_gamma]);
                mov(reg_diff_beta, EVEX_compress_addr(rsp, diff_beta));
                vmovups(zdiff_beta, zword[reg_diff_beta]);
                vmulps(zdiff_gamma, zdiff_gamma, zsqrtvar);

                uni_vbroadcastss(bwd_chan, EVEX_compress_addr(rsp, bwd_chan_size));
                vdivps(zdiff_beta, zdiff_beta, bwd_chan);
                vdivps(zdiff_gamma, zdiff_gamma, bwd_chan);

                L(no_OC_FIRST);
            }

            if (one_of(jcp.prop_kind, forward_training)) {
              Label no_OC_FIRST;
              Label reduce_loop_done;
              mov(reg_coff2, EVEX_compress_addr(rsp, coff));
              cmp(reg_coff2, 0);
              jne(no_OC_FIRST, T_NEAR);
              reduce_loop_OC_FIRST(load_loop_blk, jcp.ur_tail, 0, true);
              jmp(reduce_loop_done, T_NEAR);

              L(no_OC_FIRST);
              reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
              L(reduce_loop_done);
            }
            else if ((jcp.prop_kind == backward_data) &&
                    (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
                Label no_OC_FIRST;
                Label reduce_loop_done;
                mov(reg_coff2, EVEX_compress_addr(rsp, coff));
                cmp(reg_coff2, 0);
                jne(no_OC_FIRST, T_NEAR);
                reduce_loop_OC_FIRST(load_loop_blk, jcp.ur_tail, 0, true);
                jmp(reduce_loop_done, T_NEAR);

                L(no_OC_FIRST);
                reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
                L(reduce_loop_done);
            }
            else
                reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);

            L(bcast_loop_tail_out);
        }
    }
}
// We changed the reduce_loop function to calculate the mean/variance for BatchNorm 
// when loading the each output channel for the first time.
void jit_avx512_common_1x1_conv_kernel::reduce_loop_OC_FIRST(int load_loop_blk,
         int ur, int substep, bool wraparound)
{
    auto vreg_load = [=](int i_load, int i_fma) {
        return Zmm(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return Zmm(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](int i_load) {
        mov(reg_bias_data, EVEX_compress_addr(rsp, bias_data));
        return EVEX_compress_addr(reg_bias_data,
                                  jcp.typesize_out * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            offt = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
                    : i_ur * jcp.reduce_loop_unroll + i_reduce;
        } else {
            if (jcp.transpose_src) {
                const int reduce_group = i_reduce / 4;
                const int reduce_shift = i_reduce % 4;
                offt = 4 * (reduce_group * jcp.ic_block + i_ur) + reduce_shift;
            }
            else
                offt = i_reduce * jcp.ic_block + i_ur;
        }
        if ((jcp.prop_kind == forward_training) && (i_reduce == 0) && (jcp.reduce_dim < 1024)) {
            const unsigned char _cmp_lt_os = 1;
            Zmm z = Zmm(27);
            vmovntdqa(z, EVEX_compress_addr(reg_prev_src, jcp.typesize_in * offt, false));
            vsubps(z, z, zmean);
            vmulps(z, z, zsqrtvar);
            vfmadd213ps(z, zgamma, zbeta);
            vcmpps(vmask, z, zmm_zero, _cmp_lt_os);
            vmulps(z | vmask, z, zmm_zero);
            vmovups(EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt, false), z);
        }

       if ((jcp.prop_kind == backward_data) &&
               (i_reduce == 0) && (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
           mov(reg_diff_src, EVEX_compress_addr(rsp, diff_src));
           vmovntdqa(z, EVEX_compress_addr(reg_diff_src, jcp.typesize_in * offt, false));
           vsubps(z, z, zdiff_beta);
           mov(reg_conv_dst, EVEX_compress_addr(rsp, conv_dst));
           vmovntdqa(t, EVEX_compress_addr(reg_conv_dst, jcp.typesize_in * offt, false));
           vsubps(t, zmean, t);
           vmulps(t, t, zdiff_gamma);
           vaddps(z, z, t);
           vmulps(z, z, zsqrtvar);
           vmulps(z, z, zgamma);
           vmovups(EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt, false), z);
       }
       return EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt,
               bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        int offt;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        if (jcp.prop_kind == backward_data && jcp.ver == ver_4vnni)
            offt = (i_load * jcp.reduce_block + u0) * jcp.load_block;
        else
            offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        return EVEX_compress_addr(aux_reg_load_data,
                                  u1 * jcp.reduce_loop_load_step
                                  + jcp.typesize_in * offt);
    };

    auto output_ptr = [=](int i_load, int i_ur) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data))
            return EVEX_compress_addr(aux_reg_output_data,
                    (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                    * jcp.typesize_out);
        else
            return ptr[aux_reg_output_data +
                       (i_load
                            ? reg_output_stride * i_load
                            : 0) // TODO: Xbyak should allow 0 scale
                       + jcp.typesize_out * jcp.load_block * i_ur];
    };

    auto relu_src_ptr = [=](int i_load, int i_ur) {
        mov(aux_reg_relu_src, EVEX_compress_addr(rsp, aux_relu_src));
        return EVEX_compress_addr(aux_reg_relu_src,
                (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                * jcp.typesize_out);
    };
    auto bn_src_ptr = [=](int i_load, int i_ur) {
        mov(aux_reg_bn_src, EVEX_compress_addr(rsp, aux_bn_src));
        return EVEX_compress_addr(aux_reg_bn_src,
                (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                * jcp.typesize_out);
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_sum) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    mic_prefetcht1(output_ptr(i_load, i_ur));
                }
            }
        }

        if (jcp.with_bias
            && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_FIRST);
            jz(init_zero, T_NEAR);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur)
                    vmovups(vreg_accum(i_load, i_ur), bias_ptr(i_load));
            jmp(init_done, T_NEAR);
        }

        L(init_zero);
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxord(r, r, r);
            }
        L(init_done);
    };

    auto vcmp = [=](Xbyak::Opmask kmask,
        Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2, const unsigned char cmp) {
        if (jcp.ver == ver_4vnni)
            vpcmpd(kmask, zmm_src1, zmm_src2, cmp);
        else
            vcmpps(kmask, zmm_src1, zmm_src2, cmp);
    };

    auto vmul = [=](Xbyak::Zmm zmm_dst, Xbyak::Opmask kmask,
                     Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2) {
        if (jcp.ver == ver_4vnni)
            vpmulld(zmm_dst | kmask, zmm_src1, zmm_src2);
        else
            vmulps(zmm_dst | kmask, zmm_src1, zmm_src2);
    };

    auto vadd = [=](const Xmm& x1, const Xmm& x2, const Operand& op) {
        if (jcp.ver == ver_4vnni)
            vpaddd(x1, x2, op);
        else
            vaddps(x1, x2, op);
    };

    auto store = [=]() {

        Label store_noadd;
        if (!jcp.with_sum) {
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum(i_load, i_ur);
                vadd(r, r, output_ptr(i_load, i_ur));
            }

        L(store_noadd);

        if (jcp.with_relu) {
            const unsigned char _cmp_lt_os = 1;
            assert(ur * load_loop_blk < 30);

            Label store_norelu;
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            vpxord(zmm_zero, zmm_zero, zmm_zero);
            if (jcp.relu_negative_slope == 0) {
                zmm_relu_ns = zmm_zero;
            } else {
                mov(imm_addr64, float2int((float)jcp.relu_negative_slope));
                vmovq(xmm_relu_ns, imm_addr64);
                vbroadcastss(zmm_relu_ns, xmm_relu_ns);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    vcmp(vmask, vreg_accum(i_load, i_ur), zmm_zero,
                        _cmp_lt_os);
                    vmul(vreg_accum(i_load, i_ur), vmask,
                        vreg_accum(i_load, i_ur), zmm_relu_ns);
            }
            L(store_norelu);
        }

        if (one_of(jcp.prop_kind, forward_training))
        {
            Label store_norelu;
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            Zmm z0 = Zmm(28);  Zmm z2 = Zmm(26);
            Zmm z1 = Zmm(29);  Zmm z3 = Zmm(27);

            mov(reg_coff2, EVEX_compress_addr(rsp, coff));
            mov(reg_rbuf1, EVEX_compress_addr(rsp, rbuf1));
            mov(reg_rbuf2, EVEX_compress_addr(rsp, rbuf2));

            int r_ur = ur % 2;
            int t_ur = ur - r_ur;

            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                uni_vmovups(z0, zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64]);
                uni_vmovups(z1, zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64]);
                uni_vpxor(z2, z2, z2);
                uni_vpxor(z3, z3, z3);

                for (int i_ur = 0; i_ur < t_ur; i_ur += 2){
                    uni_vaddps(z0, z0, vreg_accum(i_load, i_ur));
                    uni_vfmadd231ps(z1, vreg_accum(i_load, i_ur), vreg_accum(i_load, i_ur));
                    uni_vaddps(z2, z2, vreg_accum(i_load, i_ur+1));
                    uni_vfmadd231ps(z3, vreg_accum(i_load, i_ur+1),
                            vreg_accum(i_load, i_ur+1));
                }

                if (r_ur) {
                    uni_vaddps(z0, z0, vreg_accum(i_load, t_ur));
                    uni_vfmadd231ps(z1, vreg_accum(i_load, t_ur), vreg_accum(i_load, t_ur));

                }
                uni_vaddps(z0, z0, z2);
                uni_vaddps(z1, z1, z3);
                uni_vmovups(zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64], z0);
                uni_vmovups(zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64], z1);
            }

            L(store_norelu);
        }

        if (one_of(jcp.prop_kind, backward_data)) {
            const unsigned char _cmp_lt_os = 0;
            Label store_norelu;
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            vpxord(zmm_zero, zmm_zero, zmm_zero);

            Zmm zmm_bn_mean = Zmm(29);
            Zmm src_relu_zmm = Zmm(25);
            Zmm src_bn= Zmm(26);
            Zmm src_bn_tmp = Zmm(27);
            Zmm z0 = Zmm(28);
            Zmm z1 = Zmm(24);

            mov(reg_coff2, EVEX_compress_addr(rsp, coff));

            mov(reg_rbuf1, EVEX_compress_addr(rsp, rbuf1));
            mov(reg_rbuf2, EVEX_compress_addr(rsp, rbuf2));

            mov(reg_bn_mean, EVEX_compress_addr(rsp, bn_mean));

            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                uni_vmovups(zmm_bn_mean, zword[reg_bn_mean + reg_coff2 * 4 + i_load * 64]);
                uni_vmovups(z0, zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64]);
                uni_vmovups(z1, zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64]);

                for (int i_ur = 0; i_ur < ur; ++i_ur){
                    vmovntdqa(src_relu_zmm, relu_src_ptr(i_load, i_ur));
                    vmovntdqa(src_bn, bn_src_ptr(i_load, i_ur));

                    vcmp(vmask, src_relu_zmm, zmm_zero, _cmp_lt_os);
                    vmul(vreg_accum(i_load, i_ur), vmask, vreg_accum(i_load, i_ur), zmm_zero);
                    vsubps(src_bn_tmp, zmm_bn_mean, src_bn);

                    if (jcp.use_vmovntps)
                        vmovntps(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
                    else
                        vmovups(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
                    vfnmadd231ps(z0, src_bn_tmp, vreg_accum(i_load, i_ur));
                    vaddps(z1, vreg_accum(i_load, i_ur));
                }
                uni_vmovups(zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64], z0);
                uni_vmovups(zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64], z1);

            }
            L(store_norelu);
        }

        if (!one_of(jcp.prop_kind, backward_data)) {
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                if (jcp.use_vmovntps)
                    vmovntps(output_ptr(i_load, i_ur),
                        vreg_accum(i_load, i_ur));
                else
                    vmovups(output_ptr(i_load, i_ur),
                        vreg_accum(i_load, i_ur));
        }
    };

    auto prefetch_callback = [=](int ur, int i_reduce, int i_ur, int i_load,
        bool last_block, bool wraparound, int reduce_step)
    {
        bool pf_ker_l1 = true;
        bool pf_ker_l2 = wraparound;
        int n_ops = (jcp.reduce_loop_unroll / reduce_step) * ur * load_loop_blk;
        int i_op = (i_reduce / reduce_step) * ur * load_loop_blk +
            i_ur * load_loop_blk + i_load;

        int n_pf_ker_l1 = pf_ker_l1 ? jcp.reduce_block : 0;
        int n_pf_ker_l2 = pf_ker_l2 && wraparound ? jcp.reduce_block : 0;
        int n_pf_out_l1 = jcp.use_vmovntps ? 0 : ur;

        int pf_inp_ops = n_ops / 2; // # of operations during which to pf input
        int pf_inp_trigger;
        if (jcp.prop_kind == backward_weights)
            pf_inp_trigger = nstl::max(1, pf_inp_ops / jcp.reduce_block);
        else
            pf_inp_trigger = nstl::max(1, pf_inp_ops / ur);

        int n_other_pf =
            load_loop_blk * (n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1);
        int n_other_pf_ops = n_ops - pf_inp_ops;
        int other_pf_trigger
                = n_other_pf ? nstl::max(1, n_other_pf_ops / n_other_pf) : 0;

        if (i_op < pf_inp_ops && i_op % pf_inp_trigger == 0) {
            // input prefetches have the highest priority b/c the
            // first iteration of the kernel block touches all the
            // cache lines
            int i_pf = i_op / pf_inp_trigger;
            auto pf_reg = wraparound && last_block
                                  ? reg_bcast_data
                                  : (last_block ? aux1_reg_bcast_data
                                                : aux_reg_bcast_data);
            int offt = i_pf;
            if (jcp.prop_kind == backward_weights) {
                offt += wraparound && last_block
                                    ? 0
                                    : (last_block ? jcp.is : jcp.reduce_block);
                offt *= jcp.bcast_block;
            } else {
                offt += wraparound && last_block
                                    ? 0
                                    : (last_block ? jcp.ur : jcp.bcast_dim);
                offt *= jcp.reduce_block;
            }
            mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
            if ((jcp.prop_kind == forward_training) && (jcp.reduce_dim < 1024)) {
                prefetcht2(ptr[reg_prev_src + offt * jcp.typesize_in]);
            }
        } else if (i_op >= pf_inp_ops && n_other_pf) {
            // remaining prefetches are spread among the rest of the
            // operations; prefetches for output take priority
            // TODO: spread L2 prefetches among L1 prefetches
            i_op -= pf_inp_ops;
            if (i_op % other_pf_trigger == 0) {
                int i_pf = i_op / (load_loop_blk * other_pf_trigger);
                if (i_pf < n_pf_ker_l2) {
                    int offt = (i_pf + (i_load + 1) * jcp.reduce_dim)
                        * jcp.load_block;
                    if (jcp.prop_kind == backward_data && jcp.ver == ver_4vnni)
                        offt = (i_pf + (i_load + 1) * jcp.reduce_block)
                                * jcp.load_block;

                    mic_prefetcht1(ptr[aux_reg_load_data
                                    + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l2 + n_pf_ker_l1) {
                    i_pf -= n_pf_ker_l2;
                    auto pf_reg = last_block ? reg_load_data
                                             : aux_reg_load_data;
                    int offt = (i_pf + i_load * jcp.reduce_dim
                        + (last_block
                            ? (wraparound ? jcp.reduce_dim : 0)
                            : jcp.reduce_block))
                        * jcp.load_block;
                    mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1) {
                    i_pf -= n_pf_ker_l1 + n_pf_ker_l2;
                    int offt = i_pf * jcp.load_block;
                    mic_prefetcht0(ptr[aux_reg_output_data
                                    + offt * jcp.typesize_out]);
                }
            }
        }
    };

    auto fma_block = [=](bool last_block) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);

        int reduce_step = jcp.fma_step;
        if (jcp.ver == ver_4vnni)
            reduce_step *= 2;

        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll;
                i_reduce += reduce_step) {
            int load_scale = (jcp.ver == ver_4vnni) ? 2 : 1;
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                // if transposed input data used and if spatial size is
                // not divided by transpose step (4) then for last reduce step
                // we should load only needed load_registers data
                // and clear remaining
                if (jcp.transpose_src && jcp.is % jcp.fma_step && last_block
                        && i_reduce == jcp.reduce_loop_unroll - reduce_step) {
                    Label load_all;
                    Label load_finish;
                    mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
                    test(reg_reduce_pos_flag2, FLAG_SP_LAST);
                    jz(load_all, T_NEAR);

                    const int n_loads = jcp.is % jcp.fma_step;
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        if (i_fma < n_loads)
                            vmovups(vreg_load(i_load, i_fma),
                                    load_ptr(i_reduce + load_scale * i_fma,
                                            i_load));
                        else
                            vpxord(vreg_load(i_load, i_fma),
                                    vreg_load(i_load, i_fma),
                                    vreg_load(i_load, i_fma));
                    }
                    jmp(load_finish);

                    L(load_all);
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        vmovups(vreg_load(i_load, i_fma),
                            load_ptr(i_reduce + load_scale * i_fma, i_load));
                    }
                    L(load_finish);
                } else {
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        vmovups(vreg_load(i_load, i_fma),
                            load_ptr(i_reduce
                                + load_scale * i_fma,
                                i_load));
                    }
                }
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    if (jcp.ver == ver_4fma)
                        v4fmaddps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, false));
                    else if (jcp.ver == ver_4vnni)
                        vp4dpwssd(vreg_accum(i_load, i_ur),
                                vreg_load(i_load, 0),
                                bcast_ptr(i_reduce, i_ur, false));
                    else
                        vfmadd231ps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, true));
                    prefetch_callback(ur, i_reduce, i_ur, i_load,
                                    last_block, wraparound, reduce_step);
                }
            }
        }
    };
    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reg_reduce_loop_work, EVEX_compress_addr(rsp, reduce_loop_work));
    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);


    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        mov(reg_scale_shift, EVEX_compress_addr(rsp, scale_shift));
        uni_vmovups(zgamma, zword[reg_scale_shift]);
        uni_vmovups(zbeta, zword[reg_scale_shift + chan_data_offt]);
    }


    if ((jcp.prop_kind == backward_data) &&
        (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
        mov(reg_conv_dst, EVEX_compress_addr(rsp, conv_dst));
        mov(EVEX_compress_addr(rsp, conv_dst_t), reg_conv_dst);
        mov(reg_diff_src, EVEX_compress_addr(rsp, diff_src));
        mov(EVEX_compress_addr(rsp, diff_src_t), reg_diff_src);

        mov(reg_next_mean, EVEX_compress_addr(rsp, next_mean));
        mov(EVEX_compress_addr(rsp, next_mean_t), reg_next_mean);
        mov(reg_next_var, EVEX_compress_addr(rsp, next_var));
        mov(EVEX_compress_addr(rsp, next_var_t), reg_next_var);
        mov(reg_next_scale_shift, EVEX_compress_addr(rsp, next_scale_shift));
        mov(EVEX_compress_addr(rsp, next_scale_shift_t), reg_next_scale_shift);
        mov(reg_diff_gamma, EVEX_compress_addr(rsp, diff_gamma));
        mov(EVEX_compress_addr(rsp, diff_gamma_t), reg_diff_gamma);
        mov(reg_diff_beta, EVEX_compress_addr(rsp, diff_beta));
        mov(EVEX_compress_addr(rsp, diff_beta_t), reg_diff_beta);
    }

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        add(reg_prev_src, jcp.reduce_loop_bcast_step);
        add(reg_prev_mean, 64);
        add(reg_prev_var, 64);

        uni_vmovups(zmean, zword[reg_prev_mean]);
        uni_vmovups(zsqrtvar, zword[reg_prev_var]);
        uni_vaddps(zsqrtvar, zsqrtvar, veps);
        vsqrtps(zsqrtvar, zsqrtvar);
        vdivps(zsqrtvar, vone, zsqrtvar);

        add(reg_scale_shift, 64);
        uni_vmovups(zgamma, zword[reg_scale_shift]);
        uni_vmovups(zbeta, zword[reg_scale_shift + chan_data_offt]);
    }


    if ((jcp.prop_kind == backward_data) &&
        (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
        mov(reg_conv_dst, EVEX_compress_addr(rsp, conv_dst));
        add(reg_conv_dst, jcp.reduce_loop_bcast_step);
        mov(EVEX_compress_addr(rsp, conv_dst), reg_conv_dst);
        mov(reg_diff_src, EVEX_compress_addr(rsp, diff_src));
        add(reg_diff_src, jcp.reduce_loop_bcast_step);
        mov(EVEX_compress_addr(rsp, diff_src), reg_diff_src);

        mov(reg_next_mean, EVEX_compress_addr(rsp, next_mean));
        add(reg_next_mean, 64);
        uni_vmovups(zmean, zword[reg_next_mean]);
        mov(EVEX_compress_addr(rsp, next_mean), reg_next_mean);
        mov(reg_next_var, EVEX_compress_addr(rsp, next_var));
        add(reg_next_var, 64);
        uni_vmovups(zsqrtvar, zword[reg_next_var]);
        mov(EVEX_compress_addr(rsp, next_var), reg_next_var);

        uni_vbroadcastss(bwd_eps, EVEX_compress_addr(rsp, eps));
        uni_vaddps(zsqrtvar, zsqrtvar, bwd_eps);
        vsqrtps(zsqrtvar, zsqrtvar);
        uni_vbroadcastss(bwd_one, EVEX_compress_addr(rsp, one));
        vdivps(zsqrtvar, bwd_one, zsqrtvar);

        mov(reg_next_scale_shift, EVEX_compress_addr(rsp, next_scale_shift));
        add(reg_next_scale_shift, 64);
        uni_vmovups(zgamma, zword[reg_next_scale_shift]);
        mov(EVEX_compress_addr(rsp, next_scale_shift), reg_next_scale_shift);
        mov(reg_diff_gamma, EVEX_compress_addr(rsp, diff_gamma));
        add(reg_diff_gamma, 64);
        uni_vmovups(zdiff_gamma, zword[reg_diff_gamma]);
        mov(EVEX_compress_addr(rsp, diff_gamma), reg_diff_gamma);
        mov(reg_diff_beta, EVEX_compress_addr(rsp, diff_beta));
        add(reg_diff_beta, 64);
        uni_vmovups(zdiff_beta, zword[reg_diff_beta]);
        mov(EVEX_compress_addr(rsp, diff_beta), reg_diff_beta);
        vmulps(zdiff_gamma, zdiff_gamma, zsqrtvar);

        uni_vbroadcastss(bwd_chan, EVEX_compress_addr(rsp, bwd_chan_size));
        vdivps(zdiff_beta, zdiff_beta, bwd_chan);
        vdivps(zdiff_gamma, zdiff_gamma, bwd_chan);
    }

        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);


    if ((jcp.prop_kind == backward_data) &&
        (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
        mov(reg_conv_dst ,EVEX_compress_addr(rsp, conv_dst_t));
        mov(EVEX_compress_addr(rsp, conv_dst), reg_conv_dst);
        mov(reg_diff_src ,EVEX_compress_addr(rsp, diff_src_t));
        mov(EVEX_compress_addr(rsp, diff_src), reg_diff_src);

        mov(reg_next_mean, EVEX_compress_addr(rsp, next_mean_t));
        mov(EVEX_compress_addr(rsp, next_mean), reg_next_mean);
        mov(reg_next_var, EVEX_compress_addr(rsp, next_var_t));
        mov(EVEX_compress_addr(rsp, next_var), reg_next_var);
        mov(reg_next_scale_shift, EVEX_compress_addr(rsp, next_scale_shift_t));
        mov(EVEX_compress_addr(rsp, next_scale_shift), reg_next_scale_shift);
        mov(reg_diff_gamma, EVEX_compress_addr(rsp, diff_gamma_t));
        mov(EVEX_compress_addr(rsp, diff_gamma), reg_diff_gamma);
        mov(reg_diff_beta, EVEX_compress_addr(rsp, diff_beta_t));
        mov(EVEX_compress_addr(rsp, diff_beta), reg_diff_beta);
    }

    store();
}


void jit_avx512_common_1x1_conv_kernel::reduce_loop(int load_loop_blk,
         int ur, int substep, bool wraparound)
{
    auto vreg_load = [=](int i_load, int i_fma) {
        return Zmm(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return Zmm(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](int i_load) {
        mov(reg_bias_data, EVEX_compress_addr(rsp, bias_data));
        return EVEX_compress_addr(reg_bias_data,
                                  jcp.typesize_out * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            offt = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
                    : i_ur * jcp.reduce_loop_unroll + i_reduce;
        } else {
            if (jcp.transpose_src) {
                const int reduce_group = i_reduce / 4;
                const int reduce_shift = i_reduce % 4;
                offt = 4 * (reduce_group * jcp.ic_block + i_ur) + reduce_shift;
            }
            else
                offt = i_reduce * jcp.ic_block + i_ur;
        }
        return EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt,
                                bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        int offt;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        if (jcp.prop_kind == backward_data && jcp.ver == ver_4vnni)
            offt = (i_load * jcp.reduce_block + u0) * jcp.load_block;
        else
            offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        return EVEX_compress_addr(aux_reg_load_data,
                                  u1 * jcp.reduce_loop_load_step
                                  + jcp.typesize_in * offt);
    };

    auto output_ptr = [=](int i_load, int i_ur) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data))
            return EVEX_compress_addr(aux_reg_output_data,
                    (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                    * jcp.typesize_out);
        else
            return ptr[aux_reg_output_data +
                       (i_load
                            ? reg_output_stride * i_load
                            : 0) // TODO: Xbyak should allow 0 scale
                       + jcp.typesize_out * jcp.load_block * i_ur];
    };

    auto relu_src_ptr = [=](int i_load, int i_ur) {
        mov(aux_reg_relu_src, EVEX_compress_addr(rsp, aux_relu_src));
        return EVEX_compress_addr(aux_reg_relu_src,
                (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                * jcp.typesize_out);
    };
    auto bn_src_ptr = [=](int i_load, int i_ur) {
        mov(aux_reg_bn_src, EVEX_compress_addr(rsp, aux_bn_src));
        return EVEX_compress_addr(aux_reg_bn_src,
                (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                * jcp.typesize_out);
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_sum) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    mic_prefetcht1(output_ptr(i_load, i_ur));
                }
            }
        }

        if (jcp.with_bias
            && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_FIRST);
            jz(init_zero, T_NEAR);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur)
                    vmovups(vreg_accum(i_load, i_ur), bias_ptr(i_load));
            jmp(init_done, T_NEAR);
        }

        L(init_zero);
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxord(r, r, r);
            }
        L(init_done);
    };

    auto vcmp = [=](Xbyak::Opmask kmask,
        Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2, const unsigned char cmp) {
        if (jcp.ver == ver_4vnni)
            vpcmpd(kmask, zmm_src1, zmm_src2, cmp);
        else
            vcmpps(kmask, zmm_src1, zmm_src2, cmp);
    };

    auto vmul = [=](Xbyak::Zmm zmm_dst, Xbyak::Opmask kmask,
                     Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2) {
        if (jcp.ver == ver_4vnni)
            vpmulld(zmm_dst | kmask, zmm_src1, zmm_src2);
        else
            vmulps(zmm_dst | kmask, zmm_src1, zmm_src2);
    };

    auto vadd = [=](const Xmm& x1, const Xmm& x2, const Operand& op) {
        if (jcp.ver == ver_4vnni)
            vpaddd(x1, x2, op);
        else
            vaddps(x1, x2, op);
    };

    auto store = [=]() {

        Label store_noadd;
        if (!jcp.with_sum) {
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum(i_load, i_ur);
                vadd(r, r, output_ptr(i_load, i_ur));
            }

        L(store_noadd);
        if (jcp.with_relu) {
            const unsigned char _cmp_lt_os = 1;
            assert(ur * load_loop_blk < 30);

            Label store_norelu;
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            vpxord(zmm_zero, zmm_zero, zmm_zero);
            if (jcp.relu_negative_slope == 0) {
                zmm_relu_ns = zmm_zero;
            } else {
                mov(imm_addr64, float2int((float)jcp.relu_negative_slope));
                vmovq(xmm_relu_ns, imm_addr64);
                vbroadcastss(zmm_relu_ns, xmm_relu_ns);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    vcmp(vmask, vreg_accum(i_load, i_ur), zmm_zero,
                        _cmp_lt_os);
                    vmul(vreg_accum(i_load, i_ur), vmask,
                        vreg_accum(i_load, i_ur), zmm_relu_ns);
            }
            L(store_norelu);
        }

        if (one_of(jcp.prop_kind, forward_training))
        {
            Label store_norelu;
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            Zmm z0 = Zmm(28);  Zmm z2 = Zmm(26);  Zmm z4 = Zmm(24);  Zmm z6 = Zmm(22);
            Zmm z1 = Zmm(29);  Zmm z3 = Zmm(27);  Zmm z5 = Zmm(25);  Zmm z7 = Zmm(23);

            mov(reg_coff2, EVEX_compress_addr(rsp, coff));

            mov(reg_rbuf1, EVEX_compress_addr(rsp, rbuf1));
            mov(reg_rbuf2, EVEX_compress_addr(rsp, rbuf2));

            int r_ur = ur % 4;
            int t_ur = ur - r_ur;

            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                uni_vmovups(z0, zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64]);
                uni_vmovups(z1, zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64]);
                uni_vpxor(z2, z2, z2);
                uni_vpxor(z3, z3, z3);
                uni_vpxor(z4, z4, z4);
                uni_vpxor(z5, z5, z5);
                uni_vpxor(z6, z6, z6);
                uni_vpxor(z7, z7, z7);

                for (int i_ur = 0; i_ur < t_ur; i_ur += 4){
                    uni_vaddps(z0, z0, vreg_accum(i_load, i_ur));
                    uni_vfmadd231ps(z1, vreg_accum(i_load, i_ur), vreg_accum(i_load, i_ur));
                    uni_vaddps(z2, z2, vreg_accum(i_load, i_ur+1));
                    uni_vfmadd231ps(z3, vreg_accum(i_load, i_ur+1),
                            vreg_accum(i_load, i_ur+1));
                    uni_vaddps(z4, z4, vreg_accum(i_load, i_ur+2));
                    uni_vfmadd231ps(z5, vreg_accum(i_load, i_ur+2),
                            vreg_accum(i_load, i_ur+2));
                    uni_vaddps(z6, z6, vreg_accum(i_load, i_ur+3));
                    uni_vfmadd231ps(z7, vreg_accum(i_load, i_ur+3),
                            vreg_accum(i_load, i_ur+3));
                }

                for(int i_ur = t_ur; i_ur < ur; i_ur++) {
                    int addr1 = 28 - 2 * (i_ur % t_ur);
                    int addr2 = 29 - 2 * (i_ur % t_ur);
                    uni_vaddps(Zmm(addr1), Zmm(addr1), vreg_accum(i_load, i_ur));
                    uni_vfmadd231ps(Zmm(addr2), vreg_accum(i_load, i_ur),
                            vreg_accum(i_load, i_ur));
                }
                uni_vaddps(z0, z0, z2);
                uni_vaddps(z0, z0, z4);
                uni_vaddps(z0, z0, z6);
                uni_vaddps(z1, z1, z3);
                uni_vaddps(z1, z1, z5);
                uni_vaddps(z1, z1, z7);
                uni_vmovups(zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64], z0);
                uni_vmovups(zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64], z1);
            }

            L(store_norelu);
        }
        if (one_of(jcp.prop_kind, backward_data)) {
            const unsigned char _cmp_lt_os = 0;
            Label store_norelu;
            mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
            test(reg_reduce_pos_flag2, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            vpxord(zmm_zero, zmm_zero, zmm_zero);

            Zmm zmm_bn_mean = Zmm(29);
            Zmm src_relu_zmm = Zmm(25);
            Zmm src_bn= Zmm(26);
            Zmm src_bn_tmp = Zmm(27);
            Zmm z0 = Zmm(28);
            Zmm z1 = Zmm(24);


            mov(reg_coff2, EVEX_compress_addr(rsp, coff));

            mov(reg_rbuf1, EVEX_compress_addr(rsp, rbuf1));
            mov(reg_rbuf2, EVEX_compress_addr(rsp, rbuf2));

            mov(reg_bn_mean, EVEX_compress_addr(rsp, bn_mean));


            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                uni_vmovups(zmm_bn_mean, zword[reg_bn_mean + reg_coff2 * 4 + i_load * 64]);
                uni_vmovups(z0, zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64]);
                uni_vmovups(z1, zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64]);

                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    vmovntdqa(src_relu_zmm, relu_src_ptr(i_load, i_ur));
                    vmovntdqa(src_bn, bn_src_ptr(i_load, i_ur));

                    vcmp(vmask, src_relu_zmm, zmm_zero, _cmp_lt_os);
                    vmul(vreg_accum(i_load, i_ur), vmask, vreg_accum(i_load, i_ur), zmm_zero);
                    vsubps(src_bn_tmp, zmm_bn_mean, src_bn);

                    if (jcp.use_vmovntps)
                        vmovntps(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
                    else
                        vmovups(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
                    vfnmadd231ps(z0, src_bn_tmp, vreg_accum(i_load, i_ur));
                    vaddps(z1, vreg_accum(i_load, i_ur));
                }
                uni_vmovups(zword[reg_rbuf1 + reg_coff2 * 4 + i_load * 64], z0);
                uni_vmovups(zword[reg_rbuf2 + reg_coff2 * 4 + i_load * 64], z1);
            }
            L(store_norelu);
        }

        if (!one_of(jcp.prop_kind, backward_data)) {
            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                    if (jcp.use_vmovntps)
                        vmovntps(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
                    else
                        vmovups(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
        }
    };

    auto prefetch_callback = [=](int ur, int i_reduce, int i_ur, int i_load,
        bool last_block, bool wraparound, int reduce_step)
    {
        bool pf_ker_l1 = true;
        bool pf_ker_l2 = wraparound;
        int n_ops = (jcp.reduce_loop_unroll / reduce_step) * ur * load_loop_blk;
        int i_op = (i_reduce / reduce_step) * ur * load_loop_blk +
            i_ur * load_loop_blk + i_load;

        int n_pf_ker_l1 = pf_ker_l1 ? jcp.reduce_block : 0;
        int n_pf_ker_l2 = pf_ker_l2 && wraparound ? jcp.reduce_block : 0;
        int n_pf_out_l1 = jcp.use_vmovntps ? 0 : ur;

        int pf_inp_ops = n_ops / 2; // # of operations during which to pf input
        int pf_inp_trigger;
        if (jcp.prop_kind == backward_weights)
            pf_inp_trigger = nstl::max(1, pf_inp_ops / jcp.reduce_block);
        else
            pf_inp_trigger = nstl::max(1, pf_inp_ops / ur);

        int n_other_pf =
            load_loop_blk * (n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1);
        int n_other_pf_ops = n_ops - pf_inp_ops;
        int other_pf_trigger
                = n_other_pf ? nstl::max(1, n_other_pf_ops / n_other_pf) : 0;

        if (i_op < pf_inp_ops && i_op % pf_inp_trigger == 0) {
            // input prefetches have the highest priority b/c the
            // first iteration of the kernel block touches all the
            // cache lines
            int i_pf = i_op / pf_inp_trigger;
            auto pf_reg = wraparound && last_block
                                  ? reg_bcast_data
                                  : (last_block ? aux1_reg_bcast_data
                                                : aux_reg_bcast_data);
            int offt = i_pf;
            if (jcp.prop_kind == backward_weights) {
                offt += wraparound && last_block
                                    ? 0
                                    : (last_block ? jcp.is : jcp.reduce_block);
                offt *= jcp.bcast_block;
            } else {
                offt += wraparound && last_block
                                    ? 0
                                    : (last_block ? jcp.ur : jcp.bcast_dim);
                offt *= jcp.reduce_block;
            }
            mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
        } else if (i_op >= pf_inp_ops && n_other_pf) {
            // remaining prefetches are spread among the rest of the
            // operations; prefetches for output take priority
            // TODO: spread L2 prefetches among L1 prefetches
            i_op -= pf_inp_ops;
            if (i_op % other_pf_trigger == 0) {
                int i_pf = i_op / (load_loop_blk * other_pf_trigger);
                if (i_pf < n_pf_ker_l2) {
                    int offt = (i_pf + (i_load + 1) * jcp.reduce_dim)
                        * jcp.load_block;
                    if (jcp.prop_kind == backward_data && jcp.ver == ver_4vnni)
                        offt = (i_pf + (i_load + 1) * jcp.reduce_block)
                                * jcp.load_block;

                    mic_prefetcht1(ptr[aux_reg_load_data
                                    + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l2 + n_pf_ker_l1) {
                    i_pf -= n_pf_ker_l2;
                    auto pf_reg = last_block ? reg_load_data
                                             : aux_reg_load_data;
                    int offt = (i_pf + i_load * jcp.reduce_dim
                        + (last_block
                            ? (wraparound ? jcp.reduce_dim : 0)
                            : jcp.reduce_block))
                        * jcp.load_block;
                    mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1) {
                    i_pf -= n_pf_ker_l1 + n_pf_ker_l2;
                    int offt = i_pf * jcp.load_block;
                    mic_prefetcht0(ptr[aux_reg_output_data
                                    + offt * jcp.typesize_out]);
                }
            }
        }
    };

    auto fma_block = [=](bool last_block) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);

        int reduce_step = jcp.fma_step;
        if (jcp.ver == ver_4vnni)
            reduce_step *= 2;

        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll;
                i_reduce += reduce_step) {
            int load_scale = (jcp.ver == ver_4vnni) ? 2 : 1;
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                // if transposed input data used and if spatial size is
                // not divided by transpose step (4) then for last reduce step
                // we should load only needed load_registers data
                // and clear remaining
                if (jcp.transpose_src && jcp.is % jcp.fma_step && last_block
                        && i_reduce == jcp.reduce_loop_unroll - reduce_step) {
                    Label load_all;
                    Label load_finish;
                    mov(reg_reduce_pos_flag2, EVEX_compress_addr(rsp, reduce_pos_flag));
                    test(reg_reduce_pos_flag2, FLAG_SP_LAST);
                    jz(load_all, T_NEAR);

                    const int n_loads = jcp.is % jcp.fma_step;
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        if (i_fma < n_loads)
                            vmovups(vreg_load(i_load, i_fma),
                                    load_ptr(i_reduce + load_scale * i_fma,
                                            i_load));
                        else
                            vpxord(vreg_load(i_load, i_fma),
                                    vreg_load(i_load, i_fma),
                                    vreg_load(i_load, i_fma));
                    }
                    jmp(load_finish);

                    L(load_all);
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        vmovups(vreg_load(i_load, i_fma),
                            load_ptr(i_reduce + load_scale * i_fma, i_load));
                    }
                    L(load_finish);
                } else {
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        vmovups(vreg_load(i_load, i_fma),
                            load_ptr(i_reduce
                                + load_scale * i_fma,
                                i_load));
                    }
                }
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    if (jcp.ver == ver_4fma)
                        v4fmaddps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, false));
                    else if (jcp.ver == ver_4vnni)
                        vp4dpwssd(vreg_accum(i_load, i_ur),
                                vreg_load(i_load, 0),
                                bcast_ptr(i_reduce, i_ur, false));
                    else
                        vfmadd231ps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, true));
                    prefetch_callback(ur, i_reduce, i_ur, i_load,
                                    last_block, wraparound, reduce_step);
                }
            }
        }
    };
    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reg_reduce_loop_work, EVEX_compress_addr(rsp, reduce_loop_work));
    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_avx512_common_1x1_conv_kernel::generate()
{
    preamble();

    chan_data_offt = jcp.ic * sizeof(float);

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);

    sub(rsp, stack_space_needed);

    mov(reg_output_data_tmp, ptr[param1 + GET_OFF(output_data)]);
    mov(EVEX_compress_addr(rsp, output_data), reg_output_data_tmp);

    if (jcp.with_bias) {
        mov(reg_bias_data_tmp, ptr[param1 + GET_OFF(bias_data)]);
        mov(EVEX_compress_addr(rsp, bias_data), reg_bias_data_tmp);
    }

    mov(reg_load_loop_work_tmp, ptr[param1 + GET_OFF(load_dim)]);
    mov(EVEX_compress_addr(rsp, load_loop_work), reg_load_loop_work_tmp);

    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_offt), reg_bcast_loop_work);

    mov(reg_reduce_loop_work_tmp, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(EVEX_compress_addr(rsp, reduce_loop_work), reg_reduce_loop_work_tmp);

    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(reduce_pos_flag)]);
    mov(EVEX_compress_addr(rsp, reduce_pos_flag), reg_reduce_pos_flag);

    if (one_of(jcp.prop_kind, forward_training))
    {

        mov(reg_prev_mean_tmp, ptr[param1 + GET_OFF(prev_mean)]);
        mov(EVEX_compress_addr(rsp, prev_mean), reg_prev_mean_tmp);

        mov(reg_prev_var_tmp, ptr[param1 + GET_OFF(prev_var)]);
        mov(EVEX_compress_addr(rsp, prev_var), reg_prev_var_tmp);

        mov(reg_prev_src_tmp, ptr[param1 + GET_OFF(prev_src)]);
        mov(EVEX_compress_addr(rsp, prev_src), reg_prev_src_tmp);

        uni_vbroadcastss(vone, zword[param1 + GET_OFF(one)]);

        uni_vbroadcastss(veps, zword[param1 + GET_OFF(eps)]);

        mov(reg_scale_shift_tmp, ptr[param1 + GET_OFF(scale_shift)]);
        mov(EVEX_compress_addr(rsp, scale_shift), reg_scale_shift_tmp);

        mov(reg_bn_src_tmp, ptr[param1 + GET_OFF(bn_src)]);
        mov(EVEX_compress_addr(rsp, bn_src), reg_bn_src_tmp);

    }

    if ((jcp.prop_kind == backward_data) &&
            (jcp.reduce_dim < 256) && !(jcp.reduce_dim == 128 && jcp.ic == 256 && jcp.oh == 57)) {
        mov(reg_conv_dst_tmp, ptr[param1 + GET_OFF(conv_dst)]);
        mov(EVEX_compress_addr(rsp, conv_dst), reg_conv_dst_tmp);
        mov(reg_diff_src_tmp, ptr[param1 + GET_OFF(diff_src)]);
        mov(EVEX_compress_addr(rsp, diff_src), reg_diff_src_tmp);

        mov(reg_next_mean_tmp, ptr[param1 + GET_OFF(next_mean)]);
        mov(EVEX_compress_addr(rsp, next_mean), reg_next_mean_tmp);
        mov(reg_next_var_tmp, ptr[param1 + GET_OFF(next_var)]);
        mov(EVEX_compress_addr(rsp, next_var), reg_next_var_tmp);
        mov(reg_next_scale_shift_tmp, ptr[param1 + GET_OFF(next_scale_shift)]);
        mov(EVEX_compress_addr(rsp, next_scale_shift), reg_next_scale_shift_tmp);
        mov(reg_diff_gamma_tmp, ptr[param1 + GET_OFF(diff_gamma)]);
        mov(EVEX_compress_addr(rsp, diff_gamma), reg_diff_gamma_tmp);
        mov(reg_diff_beta_tmp, ptr[param1 + GET_OFF(diff_beta)]);
        mov(EVEX_compress_addr(rsp, diff_beta), reg_diff_beta_tmp);

        mov(reg_one_tmp, ptr[param1 + GET_OFF(one)]);
        mov(EVEX_compress_addr(rsp, one), reg_one_tmp);
        mov(reg_eps_tmp, ptr[param1 + GET_OFF(eps)]);
        mov(EVEX_compress_addr(rsp, eps), reg_eps_tmp);
        mov(reg_bwd_chan_size_tmp, ptr[param1 + GET_OFF(bwd_chan_size)]);
        mov(EVEX_compress_addr(rsp, bwd_chan_size), reg_bwd_chan_size_tmp);

    }

    if (one_of(jcp.prop_kind, backward_data))
    {
         vmovq(xtmp, ptr[param1 + GET_OFF(relu_src)]);
         vmovq(xtmp2, ptr[param1 + GET_OFF(bn_src)]);

         mov(reg_bn_var_tmp, ptr[param1 + GET_OFF(bn_var)]);
         mov(EVEX_compress_addr(rsp, bn_var), reg_bn_var_tmp);

         mov(reg_one_tmp, ptr[param1 + GET_OFF(one)]);
         mov(EVEX_compress_addr(rsp, one), reg_one_tmp);

         mov(reg_eps_tmp, ptr[param1 + GET_OFF(eps)]);
         mov(EVEX_compress_addr(rsp, eps), reg_eps_tmp);
    }
    if(one_of(jcp.prop_kind, forward_training, backward_data))
    {
        mov(reg_rbuf1_tmp, ptr[param1 + GET_OFF(rbuf1)]);
        mov(EVEX_compress_addr(rsp, rbuf1), reg_rbuf1_tmp);
        mov(reg_rbuf2_tmp, ptr[param1 + GET_OFF(rbuf2)]);
        mov(EVEX_compress_addr(rsp, rbuf2), reg_rbuf2_tmp);

        mov(reg_bn_mean_tmp, ptr[param1 + GET_OFF(bn_mean)]);
        mov(EVEX_compress_addr(rsp, bn_mean), reg_bn_mean_tmp);

        mov(reg_last_flag_tmp, ptr[param1 + GET_OFF(last_flag)]);
        mov(EVEX_compress_addr(rsp, last_flag), reg_last_flag_tmp);

        mov(reg_rbuf1_base_tmp, ptr[param1 + GET_OFF(rbuf1_base)]);
        mov(EVEX_compress_addr(rsp, rbuf1_base), reg_rbuf1_base_tmp);
        mov(reg_rbuf2_base_tmp, ptr[param1 + GET_OFF(rbuf2_base)]);
        mov(EVEX_compress_addr(rsp, rbuf2_base), reg_rbuf2_base_tmp);

        mov(reg_mean_fusion_tmp, ptr[param1 + GET_OFF(mean_fusion)]);
        mov(EVEX_compress_addr(rsp, mean_fusion), reg_mean_fusion_tmp);

        mov(reg_var_fusion_tmp, ptr[param1 + GET_OFF(var_fusion)]);
        mov(EVEX_compress_addr(rsp, var_fusion), reg_var_fusion_tmp);

        mov(reg_ithr_tmp, ptr[param1 + GET_OFF(ithr)]);
        mov(EVEX_compress_addr(rsp, ithr), reg_ithr_tmp);

        mov(reg_nthr_tmp, ptr[param1 + GET_OFF(nthr)]);
        mov(EVEX_compress_addr(rsp, nthr), reg_nthr_tmp);

        mov(reg_chan_size_tmp, ptr[param1 + GET_OFF(chan_size)]);
        mov(EVEX_compress_addr(rsp, chan_size), reg_chan_size_tmp);

        mov(reg_barrier_tmp, ptr[param1 + GET_OFF(barrier)]);
        mov(EVEX_compress_addr(rsp, barrier), reg_barrier_tmp);

        mov(reg_coff_max_tmp, ptr[param1 + GET_OFF(coff_max)]);
        mov(EVEX_compress_addr(rsp, coff_max), reg_coff_max_tmp);

        mov(reg_base_coff_tmp, ptr[param1 + GET_OFF(base_coff)]);
        mov(EVEX_compress_addr(rsp, base_coff), reg_base_coff_tmp);
    }

    xor_(reg_coff, reg_coff);
    mov(EVEX_compress_addr(rsp, coff), reg_coff);

    if (one_of(jcp.prop_kind, forward_training, forward_inference))
        mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.relu_negative_slope));
    if (jcp.prop_kind == backward_weights)
        mov(reg_output_stride, ptr[param1 + GET_OFF(output_stride)]);

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        switch (jcp.prop_kind) {
        case forward_training:
        case forward_inference:
            mov(reg_bias_data, EVEX_compress_addr(rsp, bias_data));
            add(reg_bias_data,
                load_loop_blk * jcp.load_block * jcp.typesize_out);
            mov(EVEX_compress_addr(rsp, bias_data), reg_bias_data);
            mov(reg_output_data, EVEX_compress_addr(rsp, output_data));
            add(reg_output_data,
                load_loop_blk * jcp.bcast_dim * jcp.load_block *
                    jcp.typesize_out);
            mov(EVEX_compress_addr(rsp, output_data), reg_output_data);
            break;
        case backward_data:
            mov(reg_output_data, EVEX_compress_addr(rsp, output_data));
            add(reg_output_data,
                load_loop_blk * jcp.bcast_dim * jcp.load_block *
                    jcp.typesize_out);
            mov(EVEX_compress_addr(rsp, output_data), reg_output_data);
            vmovq(reg_relu_src, xtmp);
            add(reg_relu_src, load_loop_blk * jcp.bcast_dim * jcp.load_block *
                    jcp.typesize_out);
            vmovq(xtmp, reg_relu_src);
            vmovq(reg_bn_src, xtmp2); //EVEX_compress_addr(rsp, bn_src));
            add(reg_bn_src,
                load_loop_blk * jcp.bcast_dim * jcp.load_block *
                    jcp.typesize_out);
            vmovq(xtmp2, reg_bn_src);

            break;
        case backward_weights:
            for (int i_load = 0; i_load < load_loop_blk; i_load++)
            {
                mov(reg_output_data, EVEX_compress_addr(rsp, output_data));
                add(reg_output_data, reg_output_stride);
                mov(EVEX_compress_addr(rsp, output_data), reg_output_data);
            }
            break;
        default:
            assert(!"invalid prop_kind");
        }
        mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        mov(EVEX_compress_addr(rsp, load_loop_work), reg_load_loop_work);
        mov(reg_coff2, EVEX_compress_addr(rsp, coff));
        add(reg_coff2, load_loop_blk * jcp.load_loop_iter_step);
        mov(EVEX_compress_addr(rsp, coff), reg_coff2);
    };

    const int simd_w = 16;

    Label load_loop_blk[7];

    // changed a unrolling factor to get registers for BN recombination.
    static const int ur_cases_fma[] = {2, 4, 5, 8, 14, 19};
    static const int ur_cases_4fma[] = {2, 4, 6, 12, 28};

    const int *ur_cases = (jcp.ver == ver_4fma || jcp.ver == ver_4vnni)
        ? ur_cases_4fma : ur_cases_fma;
    const int num_ur_cases
        = (jcp.ver == ver_4fma || jcp.ver == ver_4vnni
            ? sizeof(ur_cases_4fma) : sizeof(ur_cases_fma)) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[num_ur_cases], T_NEAR);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
                    cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    je(load_loop_blk[label_idx - 1], T_NEAR);
                }
                mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
                cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                jge(load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
                cmp(reg_load_loop_work, simd_w * (idx + 1));
                je(load_loop_blk[idx], T_NEAR);
            }
            if (ur_idx < num_ur_cases - 2) {
                mov(reg_load_loop_work, EVEX_compress_addr(rsp, load_loop_work));
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);


    if (one_of(jcp.prop_kind, forward_training)) {

        Label no_last;
        mov(reg_last_flag, EVEX_compress_addr(rsp, last_flag));
        test(reg_last_flag, FLAG_LAST);
        jz(no_last, T_NEAR);

        mov(reg_rbuf1_base, EVEX_compress_addr(rsp, rbuf1_base));
        mov(reg_rbuf2_base, EVEX_compress_addr(rsp, rbuf2_base));
        mov(reg_mean_fusion, EVEX_compress_addr(rsp, mean_fusion));
        mov(reg_var_fusion, EVEX_compress_addr(rsp, var_fusion));
        mov(reg_coff_max, EVEX_compress_addr(rsp, coff_max));
        mov(reg_nthr, EVEX_compress_addr(rsp, nthr));
        mov(reg_ithr, EVEX_compress_addr(rsp, ithr));
        mov(reg_chan_size, EVEX_compress_addr(rsp, chan_size));

        Label no_mean_reduction;
        mov(reg_barrier, EVEX_compress_addr(rsp, barrier));
        simple_barrier::generate(*this, reg_barrier, reg_nthr);
        {
            cmp(reg_ithr, 0);
            jne(no_mean_reduction);
            mov(reg_base_coff, EVEX_compress_addr(rsp, base_coff));
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                mov(reg_roff, reg_base_coff);
                uni_vpxor(Zmm(21), Zmm(21), Zmm(21));
                uni_vpxor(Zmm(22), Zmm(22), Zmm(22));
                mov(reg_ctr, reg_nthr);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs); {
                    uni_vaddps(Zmm(21), Zmm(21), zword[reg_rbuf1_base + reg_roff * 4]);
                    uni_vaddps(Zmm(22), Zmm(22), zword[reg_rbuf2_base + reg_roff * 4]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(mean_reduction_thrs);
                }
                uni_vdivps(Zmm(21), Zmm(21), zword[reg_chan_size]);
                uni_vmovups(zword[reg_mean_fusion + reg_base_coff * 4], Zmm(21));
                uni_vdivps(Zmm(22), Zmm(22), zword[reg_chan_size]);
                uni_vpxor(Zmm(23), Zmm(23), Zmm(23));
                uni_vfmadd231ps(Zmm(23), Zmm(21), Zmm(21));
                vsubps(Zmm(22), Zmm(22), Zmm(23));
                uni_vmovups(zword[reg_var_fusion + reg_base_coff * 4], Zmm(22));
                add(reg_base_coff, 16);
                cmp(reg_base_coff, reg_coff_max);
                jne(mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        L(no_last);
    }

    if (one_of(jcp.prop_kind, backward_data)) {

        Label no_last;
        mov(reg_last_flag, EVEX_compress_addr(rsp, last_flag));
        test(reg_last_flag, FLAG_LAST);
        jz(no_last, T_NEAR);

        uni_vbroadcastss(vone, EVEX_compress_addr(rsp, one));// zword[reg_one]);
        uni_vbroadcastss(veps, EVEX_compress_addr(rsp, eps));

        mov(reg_rbuf1_base, EVEX_compress_addr(rsp, rbuf1_base));
        mov(reg_rbuf2_base, EVEX_compress_addr(rsp, rbuf2_base));
        mov(reg_mean_fusion, EVEX_compress_addr(rsp, mean_fusion));
        mov(reg_var_fusion, EVEX_compress_addr(rsp, var_fusion));
        mov(reg_coff_max, EVEX_compress_addr(rsp, coff_max));
        mov(reg_nthr, EVEX_compress_addr(rsp, nthr));
        mov(reg_ithr, EVEX_compress_addr(rsp, ithr));
        mov(reg_chan_size, EVEX_compress_addr(rsp, chan_size));


        mov(reg_bn_var, EVEX_compress_addr(rsp, bn_var));

        Label no_mean_reduction;
        mov(reg_barrier, EVEX_compress_addr(rsp, barrier));
        simple_barrier::generate(*this, reg_barrier, reg_nthr);
        {
            cmp(reg_ithr, 0);
            jne(no_mean_reduction);
            mov(reg_base_coff, EVEX_compress_addr(rsp, base_coff));
            vmovups(zsqrtvar, zword[reg_bn_var + reg_base_coff * 4]);
            vaddps(zsqrtvar, zsqrtvar, veps);
            vsqrtps(zsqrtvar, zsqrtvar);
            vdivps(zsqrtvar, vone, zsqrtvar);
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                mov(reg_roff, reg_base_coff);
                uni_vpxor(Zmm(21), Zmm(21), Zmm(21));
                uni_vpxor(Zmm(22), Zmm(22), Zmm(22));
                mov(reg_ctr, reg_nthr);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs); {
                    uni_vaddps(Zmm(21), Zmm(21), zword[reg_rbuf1_base + reg_roff * 4]);
                    uni_vaddps(Zmm(22), Zmm(22), zword[reg_rbuf2_base + reg_roff * 4]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(mean_reduction_thrs);
                }
                vmulps(Zmm(21), Zmm(21), zsqrtvar);
                uni_vmovups(zword[reg_mean_fusion + reg_base_coff * 4], Zmm(21));
                uni_vmovups(zword[reg_var_fusion + reg_base_coff * 4], Zmm(22));
                add(reg_base_coff, 16);
                cmp(reg_base_coff, reg_coff_max);
                jne(mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        L(no_last);
    }

    add(rsp, stack_space_needed);

    postamble();
}

bool jit_avx512_common_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) {
        return p.entry_[idx].kind == eltwise
            && p.entry_[idx].eltwise.scale == 1.
            && p.entry_[idx].eltwise.alg == alg_kind::eltwise_relu
            && p.entry_[idx].eltwise.alpha == 0.;
    };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return true // sum OR relu
                && !jcp.with_relu
                && (is_relu(0) || p.contain(sum, 0));
    case 2: return true // sum->relu
                && !jcp.with_relu
                && (p.contain(sum, 0) && is_relu(1));
    default: return false;
    }

    return false;
}

status_t jit_avx512_common_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
        bool with_relu, float relu_negative_slope,
        int nthreads, bool reduce_src)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];

    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = one_of(jcp.prop_kind, forward_training, forward_inference)
        ? cd.bias_desc.format != memory_format::undef : false;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;
    jcp.tr_is = rnd_up(jcp.is, 4);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    if (!jcp.with_relu) {
        jcp.with_relu = p.find(primitive_kind::eltwise) != -1;
        jcp.relu_negative_slope = 0;
    }

    bool args_ok = true
        && jcp.ngroups == 1
        && src_d.format() == nChw16c
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == nChw16c;
    if (!args_ok) return status::unimplemented;

    const int simd_w = 16;

    args_ok = true
        && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.stride_w == 1 && jcp.stride_h == 1 // TODO: support some strides
        && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;
    jcp.transpose_src = false;

    if (mayiuse(avx512_mic_4ops)
        && ((one_of(jcp.prop_kind, forward_training, forward_inference)
            && src_d.data_type() == data_type::s16
            && weights_d.data_type() == data_type::s16
            && dst_d.data_type() == data_type::s32)
        || (jcp.prop_kind == backward_data
            && src_d.data_type() == data_type::s32
            && weights_d.data_type() == data_type::s16
            && dst_d.data_type() == data_type::s16)))
    {
        constexpr memory_format_t weights_formats[2][2] = {
            { OIhw8i16o2i, OIhw8o16i2o },
            { gOIhw8i16o2i, gOIhw8o16i2o }
        };
        memory_format_t weights_format
            = weights_formats[with_groups][jcp.prop_kind == backward_data];
        if (weights_d.format() != weights_format)
            return status::unimplemented;

        jcp.ver = ver_4vnni;
        jcp.fma_step = 4;
        jcp.typesize_in = sizeof(prec_traits<data_type::s16>::type);
        jcp.typesize_out = sizeof(prec_traits<data_type::s32>::type);
    }
    else if (everyone_is(data_type::f32, src_d.data_type(),
                            weights_d.data_type(), dst_d.data_type()))
    {
        constexpr memory_format_t weights_formats[2][2] = {
            { OIhw16i16o, IOhw16o16i },
            { gOIhw16i16o, gIOhw16o16i }
        };
        memory_format_t weights_format
            = weights_formats[with_groups][jcp.prop_kind == backward_data];

        if (weights_d.format() != weights_format)
            return status::unimplemented;
        if (jcp.prop_kind != backward_weights && mayiuse(avx512_mic_4ops) &&
            ((jcp.prop_kind == backward_data) ? jcp.oc_block : jcp.ic_block) % 4
            == 0) {
            jcp.ver = ver_4fma;
            jcp.fma_step = 4;
        } else if (jcp.prop_kind == backward_weights && mayiuse(avx512_mic_4ops)
                && !reduce_src
                /* Heuristic condition for relation of src size to oc. Otherwise
                   the src transposition overhead exceed the benefit from 4fma
                */
                && ((jcp.is * jcp.ic) / jcp.oc <= 2048)
                )
        {
            jcp.transpose_src = true;
            jcp.ver = ver_4fma;
            jcp.fma_step = 4;
        } else {
            jcp.ver = ver_fma;
            jcp.fma_step = 1;
        }
        jcp.typesize_in = sizeof(prec_traits<data_type::f32>::type);
        jcp.typesize_out = sizeof(prec_traits<data_type::f32>::type);
    } else {
        return status::unimplemented;
    }

    jcp.ur = 1;

    // changed max_regs to get registers for BN recombination.
    int max_regs = 19;
    int min_regs = 8;

    int size_treshold;
    if (jcp.ver == ver_4fma || jcp.ver == ver_4vnni)
        size_treshold = 28;
    else
        size_treshold = 14;

    int ur_step = 1;
    if (jcp.ver == ver_4fma || jcp.ver == ver_4vnni)
        ur_step = 4;
    for (int ur_w = max_regs; ur_w >= min_regs; ur_w -= ur_step) {
        if ((jcp.ih >= size_treshold && jcp.ih % ur_w == 0)
            || (jcp.ih < size_treshold && jcp.os % ur_w == 0)) {
            jcp.ur = ur_w;
            break;
        }
    }
    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

    if (jcp.ur == 1) {
        jcp.ur = nstl::min(max_regs, jcp.os);
        int os_tail = jcp.os % max_regs;
        for (int i = max_regs; i >= min_regs; i -= ur_step) {
            int i_tail = jcp.os % i;
            if (i_tail > os_tail || i_tail == 0) {
                jcp.ur = i;
                os_tail = i_tail;
                if (i_tail == 0)
                    break;
            }
        }
    }

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };
    int reduce_blocking_max{ 0 };

    jcp.load_grp_count = 1;
    jcp.use_vmovntps = true;

    const int L2_capacity = (512 * 1024 * 3) / (4 * sizeof(float));
    const int L1_capacity = (32 * 1024) / sizeof(float);

    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
            jcp.reduce_dim = jcp.ic;
            jcp.reduce_block = jcp.ic_block;

            jcp.load_dim = jcp.oc;
            jcp.load_block = jcp.oc_block;

            jcp.bcast_dim = jcp.is;
        } else {
            jcp.reduce_dim = jcp.oc;
            jcp.reduce_block = jcp.oc_block;

            jcp.load_dim = jcp.ic;
            jcp.load_block = jcp.ic_block;

            jcp.bcast_dim = jcp.os;
        }
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        if (jcp.prop_kind == backward_data && jcp.ver == ver_4vnni)
            jcp.reduce_loop_load_step
                    = jcp.reduce_loop_unroll * jcp.ic * jcp.typesize_in;
        else
            jcp.reduce_loop_load_step
                    = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

        jcp.bcast_loop_output_step = jcp.ur * jcp.load_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.reduce_block * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = -1; // unused

        if (jcp.prop_kind == backward_data && jcp.ver == ver_4vnni)
            jcp.load_loop_load_step
                    = jcp.oc_block * jcp.ic_block * jcp.typesize_in;
        else
            jcp.load_loop_load_step
                    = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

        jcp.load_loop_iter_step = jcp.load_block;

        if (jcp.prop_kind == backward_data)
            jcp.loop_order = loop_lbr;
        else
            jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

        reduce_blocking = nb_reduce;
        if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
            reduce_blocking = 16;
        else if (jcp.bcast_dim > SMALL_SPATIAL &&
                    jcp.reduce_dim >= BIG_REDUCE_DIM)
            reduce_blocking = 8;
        reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
        reduce_blocking *= jcp.reduce_block;

        // TODO
        if (0) {
            if (jcp.prop_kind == backward_data)
                jcp.loop_order = reduce_src ? loop_lbr : loop_rlb;
            else
                jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
            jcp.use_vmovntps = false;
        }

        load_blocking = jcp.load_dim;

        jcp.load_grp_count = div_up(nthreads, jcp.mb * jcp.ngroups * nb_bcast);
        jcp.load_grp_count = best_divider(
                nthreads, jcp.load_grp_count, 2 * jcp.load_grp_count, false);
        if (jcp.bcast_dim <= 49 && jcp.mb <= nthreads
            && jcp.load_dim > 512
            && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                 div_up(nthreads, jcp.load_grp_count))
                * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast
                = (L2_capacity - /* kernel_size - */
                    2 * jcp.load_block * reduce_blocking
                        - jcp.ur * reduce_blocking - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity)
            space_for_bcast /= 2;

        int bcast_in_cache
                = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

    } else if (jcp.prop_kind == backward_weights) {

        jcp.use_vmovntps = false;
        if (jcp.is > SMALL_SPATIAL && jcp.ver == ver_4fma)
            jcp.use_vmovntps = true;

        if (jcp.transpose_src)
            jcp.reduce_dim = jcp.tr_is;
        else
            jcp.reduce_dim = jcp.is;

        if (jcp.ver == ver_4fma) {
            // reduce_block should be divided by fma_step
            jcp.reduce_block = best_divider(jcp.reduce_dim, 4, 16, true, 4);
        } else {
            jcp.reduce_block = best_divider(jcp.reduce_dim, 7, 16, true);
            if (jcp.reduce_dim % jcp.reduce_block != 0)
                jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false);
        }


        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.ur = jcp.bcast_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.ic_block * jcp.typesize_in;
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * jcp.typesize_in;

        jcp.bcast_loop_output_step =
                                jcp.oc_block * jcp.ic_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep =
            jcp.oc_block * jcp.ur * jcp.typesize_out;
        jcp.bcast_loop_bcast_step =
                jcp.ic_block * jcp.reduce_dim * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = jcp.ur * jcp.typesize_in;

        jcp.load_loop_load_step = jcp.oc_block * jcp.os * jcp.typesize_in;
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */
        balance(jcp, nthreads);

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        load_blocking = best_divider(load_blocking, 16, load_blocking, false);
        load_blocking *= jcp.load_block;

        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        int max_bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        int min_bcast_blocking = 5;

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        bcast_blocking = best_divider(
                bcast_blocking, min_bcast_blocking, max_bcast_blocking, false);
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        // for reduction balance
        int max_reduce_blocking = L2_capacity /
            ((bcast_blocking + load_blocking) * jcp.reduce_block);
        max_reduce_blocking = nstl::min(max_reduce_blocking,
            (L1_capacity / (jcp.bcast_block)) / jcp.reduce_block);

        int num_jobs = div_up(jcp.load_dim, load_blocking) *
                        div_up(jcp.bcast_dim, bcast_blocking);
        int threads_per_job = nstl::max(1, nthreads / num_jobs);
        reduce_blocking = div_up(jcp.mb * jcp.reduce_dim, jcp.reduce_block);
        reduce_blocking = div_up(reduce_blocking, threads_per_job);
        reduce_blocking = best_divider(reduce_blocking,
                    max_reduce_blocking - 2, max_reduce_blocking, true);
        reduce_blocking *= jcp.reduce_block;

        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);
    if (jcp.ver == ver_4fma || jcp.ver == ver_4vnni) {
        if (jcp.ver == ver_4fma)
            assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);
        if (jcp.ver == ver_4vnni)
            assert(jcp.reduce_loop_unroll % (2 * jcp.fma_step) == 0);
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
    }

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

void jit_avx512_common_1x1_conv_kernel::balance(jit_1x1_conv_conf_t &jcp,
        int nthreads)
{
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        jcp.nthr_ = jcp.nthr_mb_ = jcp.nthr_g_ =
            jcp.nthr_oc_b_ = jcp.nthr_ic_b_ = 1;
        return;
    }
    const int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    const int nb_load = div_up(jcp.load_dim, jcp.load_block);
    const int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    jcp.nthr_g_ = jcp.ngroups;
    const int nthr = nthreads / jcp.nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
        * optimizer tries to minimize memory consumption. few notes: (n1)
        * unclear why, but that essentially helps first convolution...
        *  (n2) assuming the reduction over minibatch is always there:
        *    - instead of 8 it should be 5 here (write ~= 2 read):
        *      kernel: temporal workspace 1 write
        *      reduction: 1 read from workspace and 1 write to the diff_wei
        *    - but experiments showed 8 works better than 5 or 6... */
        int bcast_koeff = 1;
        int load_koeff = 1;
        int output_koeff = 12;
        if (jcp.transpose_src) {
            bcast_koeff = 5;
            load_koeff = 1;
            output_koeff = 8;
        }
        return 0
            + bcast_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
            * div_up(jcp.ngroups, jcp.nthr_g_)
            * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.reduce_block
            / jcp.stride_h / jcp.stride_w /* (n1) */
            + load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
            * div_up(jcp.ngroups, jcp.nthr_g_)
            * div_up(nb_load, nthr_oc_b) * jcp.oc_block * jcp.reduce_block
            + output_koeff /* (n2) */
            * div_up(jcp.ngroups, jcp.nthr_g_) * div_up(nb_load, nthr_oc_b)
            * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block
            * jcp.oc_block;
    };

    int nthr_mb = 1, nthr_oc_b = 1, nthr_ic_b = 1;
    int best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, jcp.mb * nb_reduce);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, nb_load);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, nb_bcast);
            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                jcp.nthr_mb_ = nthr_mb;
                jcp.nthr_oc_b_ = nthr_oc_b;
                jcp.nthr_ic_b_ = nthr_ic_b;
            }
        }
    }
    if (jcp.nthr_mb_ > nthreads / 2 && jcp.nthr_mb_ < nthreads)
        jcp.nthr_mb_ = nstl::min(jcp.mb, nthreads);

    jcp.nthr_ = jcp.nthr_mb_ * jcp.nthr_g_ * jcp.nthr_oc_b_ * jcp.nthr_ic_b_;
    assert(jcp.nthr_ <= nthreads);
}

}
}
}
