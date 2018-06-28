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

#ifndef JIT_AVX512_COMMON_1x1_CONV_KERNEL_HPP
#define JIT_AVX512_COMMON_1x1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
struct jit_avx512_common_1x1_conv_kernel : public jit_generator {
    jit_avx512_common_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp,
            const primitive_attr_t &attr) : jcp(ajcp), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *)) this->getCode();
    }

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
                                const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                                const convolution_desc_t &cd,
                                const memory_desc_wrapper &src_d,
                                const memory_desc_wrapper &weights_d,
                                const memory_desc_wrapper &dst_d,
                                const primitive_attr_t &attr,
                                bool with_relu, float relu_negative_slope,
                                int nthreads, bool reduce_src);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                              const convolution_desc_t &cd,
                              const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d,
                              const primitive_attr_t &attr,
                              int nthreads, bool reduce_src)
    {
        return init_conf(jcp, cd, src_d, weights_d, dst_d, attr, false, 0.0,
        nthreads, reduce_src);
    }

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

  private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;

    // We rearrange registers without conflicts for BatchNorm fusion.
    reg64_t reg_bcast_data = r8;
    reg64_t reg_load_data = r10;
    reg64_t reg_output_data = r9;
    reg64_t reg_output_data_tmp = rbx;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_load_data = r15;
    reg64_t imm_addr64 = aux_reg_load_data;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reg_load_loop_work = rsi;
    reg64_t reg_load_loop_work_tmp = rbx;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t reg_reduce_loop_work_tmp = rbx;
    reg64_t bcast_loop_iter = rdx;
    reg64_t reduce_loop_iter = abi_param1;
    reg64_t reg_reduce_pos_flag = rbx;
    reg64_t reg_coff = rbx;
    reg64_t reg_output_stride = r13;
    reg64_t reg_bias_data = r12;
    reg64_t reg_bias_data_tmp = rbx;
    reg64_t reg_relu_ns = r13; //not used
    reg64_t reg_bcast_loop_work = aux1_reg_bcast_data;
    reg64_t reg_rbuf1 = r9;
    reg64_t reg_rbuf1_tmp = rbx;
    reg64_t reg_rbuf1_base = r9;
    reg64_t reg_rbuf1_base_tmp = rbx;
    reg64_t reg_rbuf2 = r13;
    reg64_t reg_rbuf2_tmp = rbx;
    reg64_t reg_rbuf2_base = r13;
    reg64_t reg_rbuf2_base_tmp = rbx;

    // for mean/variance fusion
    reg64_t reg_mean_fusion = r8;
    reg64_t reg_mean_fusion_tmp = rbx;
    reg64_t reg_var_fusion = r10;
    reg64_t reg_var_fusion_tmp = rbx;
    reg64_t reg_ithr = rbp;
    reg64_t reg_ithr_tmp = rbx;
    reg64_t reg_nthr = r12;
    reg64_t reg_nthr_tmp = rbx;
    reg64_t reg_chan_size = r15;
    reg64_t reg_chan_size_tmp = rbx;
    reg64_t reg_coff_max = r11;
    reg64_t reg_coff_max_tmp = rbx;
    reg64_t reg_coff2 = rax;
    reg64_t reg_base_coff = rbp;
    reg64_t reg_base_coff_tmp = rbx;
    reg64_t reg_reduce_pos_flag2 = rax;
    reg64_t reg_last_flag = rax;
    reg64_t reg_last_flag_tmp = rbx;

    // norm fusion fwd
    reg64_t reg_prev_mean = r9;
    reg64_t reg_prev_mean_tmp = rbx;
    reg64_t reg_prev_var = r13;
    reg64_t reg_prev_var_tmp = rbx;
    reg64_t reg_prev_src = r12;
    reg64_t reg_prev_src_tmp = rbx;
    reg64_t reg_scale_shift = r11;
    reg64_t reg_scale_shift_tmp = rbx;

    // norm fusion bwd
    reg64_t reg_next_mean = r9;
    reg64_t reg_next_mean_tmp = rbx;
    reg64_t reg_next_var = r9;
    reg64_t reg_next_var_tmp = rbx;
    reg64_t reg_conv_dst = r9;
    reg64_t reg_conv_dst_tmp = rbx;
    reg64_t reg_diff_src = r9;
    reg64_t reg_diff_src_tmp = rbx;
    reg64_t reg_next_scale_shift = r9;
    reg64_t reg_next_scale_shift_tmp = rbx;
    reg64_t reg_diff_gamma = r9;
    reg64_t reg_diff_gamma_tmp = rbx;
    reg64_t reg_diff_beta = r9;
    reg64_t reg_diff_beta_tmp = rbx;
    reg64_t reg_bwd_chan_size = r9;
    reg64_t reg_bwd_chan_size_tmp = rbx;

    // bwd fusion
    reg64_t reg_relu_src = rbx;
    reg64_t aux_reg_relu_src = rsi;
    reg64_t reg_relu_src_tmp = rbx;
    reg64_t reg_bn_src = r9;
    reg64_t aux_reg_bn_src = r11;
    reg64_t reg_bn_src_tmp = rbx;

    reg64_t reg_bn_mean = r12;
    reg64_t reg_bn_mean_tmp = rbx;

    reg64_t reg_bn_var = rbx;
    reg64_t reg_bn_var_tmp = rbx;

    reg64_t reg_one_tmp = rbx;

    reg64_t reg_eps_tmp = rbx;

    //mask_t vmask = k7;
    Opmask vmask = Opmask(1);

    reg64_t reg_roff = r14;
    reg64_t reg_ctr = rsi;

    reg64_t reg_barrier = rax;
    reg64_t reg_barrier_tmp = rbx;

    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(30);
    Xbyak::Zmm zmm_relu_ns = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);
    Xbyak::Zmm vone = Xbyak::Zmm(30);
    Xbyak::Zmm veps = Xbyak::Zmm(22);

    Xbyak::Zmm zmean = Xbyak::Zmm(26);
    Xbyak::Zmm zsqrtvar = Xbyak::Zmm(25);
    Xbyak::Zmm zgamma = Xbyak::Zmm(24);
    Xbyak::Zmm zbeta = Xbyak::Zmm(23);

    Xbyak::Xmm xtmp = Xbyak::Xmm(21);
    Xbyak::Xmm xtmp2 = Xbyak::Xmm(30);

    Xbyak::Zmm bwd_one = Xbyak::Zmm(27);
    Xbyak::Zmm bwd_eps = Xbyak::Zmm(22);
    Xbyak::Zmm bwd_chan = Xbyak::Zmm(31);
    Xbyak::Zmm zdiff_gamma = Xbyak::Zmm(28);
    Xbyak::Zmm zdiff_beta = Xbyak::Zmm(29);
    Xbyak::Zmm z = Xbyak::Zmm(27);
    Xbyak::Zmm t = Xbyak::Zmm(22);

    int bcast_loop_work_offt = 0;
    int reduce_pos_flag = 16;
    int coff = 32;
    int output_data = 48;
    int rbuf1 = 64;
    int rbuf2 = 80;
    int mean_fusion = 96;
    int var_fusion = 112;
    int ithr = 128;
    int nthr = 144;
    int chan_size = 160;
    int last_flag = 176;
    int coff_max = 192;
    int barrier = 208;
    int base_coff = 224;
    int rbuf1_base = 240;
    int rbuf2_base = 256;
    int prev_mean = 272;
    int prev_var = 288;
    int reduce_loop_work = 304;
    int prev_src = 320;
    int scale_shift = 336;
    int bias_data = 352;
    int load_loop_work = 368;

    // bwd
    int relu_src = 384;
    int aux_relu_src = 400;
    int bn_src = 416;
    int bn_mean = 432;
    int aux_bn_src = 448;
    int bn_var = 464;
    int one = 480;
    int eps = 496;

    // bwd norm
    int next_mean = 512;
    int next_var = 528;
    int conv_dst = 544;
    int diff_src = 560;
    int next_scale_shift = 576;
    int diff_gamma = 592;
    int diff_beta = 608;
    int bwd_chan_size = 624;
    int conv_dst_t = 640;
    int diff_src_t = 656;
    int next_mean_t = 672;
    int next_var_t = 688;
    int next_scale_shift_t = 704;
    int diff_gamma_t = 720;
    int diff_beta_t = 736;

    int stack_space_needed = 752;

    int chan_data_offt;
    int diff_chan_data_offt;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);
    void reduce_loop_OC_FIRST(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    static void balance(jit_1x1_conv_conf_t &jcp, int nthreads);
};
}
}
}

#endif
