/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef JIT_PRIMITIVE_CONF_HPP
#define JIT_PRIMITIVE_CONF_HPP

#include <stdint.h>
#include "cpu_barrier.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace barrier = simple_barrier;
/* convolution */
enum conv_version_t {ver_unused, ver_fma, ver_4fma, ver_4vnni};
enum conv_loop_order_t {loop_cgn, loop_gnc, loop_ngc};
enum conv_1x1_loop_order_t {loop_rbl, loop_rlb, loop_lbr, loop_lrb, loop_blr,
                            loop_brl};

enum {
    FLAG_MB_FIRST = 1 << 0, FLAG_MB_LAST = 1 << 1,
    FLAG_OC_FIRST = 1 << 2, FLAG_OC_LAST = 1 << 3,
    FLAG_IC_FIRST = 1 << 4, FLAG_IC_LAST = 1 << 5,
    FLAG_SP_FIRST = 1 << 6, FLAG_SP_LAST = 1 << 7,
    FLAG_REDUCE_FIRST = 1<<8, FLAG_REDUCE_LAST = 1<<9,
    FLAG_LAST = 1 << 10,
    FLAG_OH_SECOND = 1 << 11, FLAG_OC_FIRST2 = 1 << 12,
};

struct jit_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;
    conv_loop_order_t loop_order;

    int mb;
    int ngroups, ic, oc;
    int ih, iw, oh, ow;
    int l_pad, t_pad;
    int r_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    float relu_negative_slope;
    bool with_sum;

    int ihp, iwp, ohp, owp;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    int nb_ic_blocking_max;
    int nb_ic_L2;
    int nb_oc_L2;
    int ur_h, ur_w;
    int ur_w_tail;
    bool is_1stconv;
    /* 4fma */
    int tr_iw;
    int tr_src_num_guard_elems;
    /* 1st conv: 4fma */
    int tr_ld;
    int kh_step;
    /* 4vnni */
    int typesize_in;
    int typesize_out;
    /* avx512_u8s8u8 */
    int ic_nb1, ic_nb2;
    int oc_nb1;
    int ur_ow_max, ur_ow, ur_ow_tail;
    int ur_ow_nsteps;
    data_type_t bia_dt;
    data_type_t dst_dt;
    int spat_size;
};


struct jit_conv_winograd_conf_t : public jit_conv_conf_t {
    //alpha determines the tile size
    static const int alpha = 6;
    int itiles;
    int jtiles;
    int ntiles;
    int ic_simd_block;
    int tile_4fma_padding;
    int tile_4fma;
    int oc_simd_block;
    int tile_block;
    int tile_block_ur;
    int nb_tile_block_ur;

    bool double_buffering;
    int zmm_start;
    int nb_reg;

    int dimK;
    int dimK_4fma;
    int dimK_reg_block;
    int dimK_block;
    int dimK_nb_block;

    int dimM;
    int dimM_simd_block;
    int dimM_block;
    int dimM_nb_block;

    int dimN;
    int dimN_reg_block;
    int dimN_block;
    int dimN_nb_block;
};

struct jit_conv_call_s {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *src_prf;
    const void *dst_prf;
    const void *filt_prf;
    const void *bias_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    size_t channel;
    size_t channel_prf;
    size_t oc_blocks;
    int flag_oc_last;
    int flags;
    int flag_last;
    int flag_oc_last_prf;
    int flag_last_prf;
    int oh_second_flags;

    const void *scale_shift;
    const void *prev_mean;
    const void *prev_var;
    const void *prev_src;
    const void *scale_shift_prf;
    const void *prev_mean_prf;
    const void *prev_var_prf;
    const void *prev_src_prf;

    float eps;
    float one;
    float eps_prf;
    float one_prf;
    float bwd_chan_size;

    // bwd store
    float *rbuf1;
    float *rbuf2;
    const void *relu_src;
    const void *bn_src;
    const void *bn_mean;
    const void *bn_var;

    float *rbuf1_prf;
    float *rbuf2_prf;
    const void *relu_src_prf;
    const void *bn_src_prf;
    const void *bn_mean_prf;

    size_t coff;
    size_t coff_prf;

    // bwd reduction
    float *rbuf1_base;
    float *rbuf2_base;
    float *diff_gamma;
    float *diff_beta;
    float *chan_size;
    size_t nthr;
    size_t ithr;
    size_t coff_max;
    size_t base_coff;
    barrier::ctx_t *barrier;

    const void *bn_var_prf;
    float *rbuf1_base_prf;
    float *rbuf2_base_prf;
    float *diff_gamma_prf;
    float *diff_beta_prf;
    float *chan_size_prf;
    size_t nthr_prf;
    size_t ithr_prf;
    size_t coff_max_prf;
    size_t base_coff_prf;
    barrier::ctx_t *barrier_prf;
};

struct jit_1x1_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, ow, oh;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    float relu_negative_slope;
    bool with_sum;

    int is, os;
    int ic_block, oc_block;

    int ur, ur_tail;

    int reduce_dim, reduce_block, nb_reduce,
        nb_reduce_blocking, nb_reduce_blocking_max;
    int load_dim, load_block, nb_load,
        nb_load_blocking, nb_load_blocking_max;
    int bcast_dim, bcast_block, nb_bcast,
        nb_bcast_blocking, nb_bcast_blocking_max;

    int reduce_loop_unroll, reduce_loop_bcast_step, reduce_loop_load_step;
    int load_loop_load_step, load_loop_iter_step;
    int bcast_loop_output_step, bcast_loop_output_substep;
    int bcast_loop_bcast_step, bcast_loop_bcast_substep;
    int fma_step;
    int load_grp_count;
    conv_1x1_loop_order_t loop_order;
    bool use_vmovntps;
    /* 4vnni */
    int typesize_in;
    int typesize_out;

    /* 4fma */
    bool transpose_src;
    int tr_is;
    int nthr_, nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_;

    int spat_size;
    int chan_data_offt;

};

struct jit_gemm_conv_conf_t {
    prop_kind_t prop_kind;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, ow, oh;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    float relu_negative_slope;

    int is, os, ks;
    int ic_block, oc_block;
    bool need_im2col;
    size_t im2col_size;
};

struct jit_1x1_conv_call_s {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *relu_src;
    const void *bn_src;
    const void *bn_mean;
    const void *bn_var;
    const void *bias_data; // used in forward and backward_weights only

    const void *scale_shift;
    const void *prev_mean;
    const void *prev_var;
    const void *prev_src;

    float eps;
    float one;
    float *rbuf1, *rbuf2;
    float *rbuf1_base, *rbuf2_base;

    float *mean_fusion;
    float *var_fusion;
    float *chan_size;
    size_t nthr;
    size_t ithr;
    size_t coff_max;
    barrier::ctx_t *barrier;

    // bwd norm fusion
    const void *next_mean;
    const void *next_var;
    const void *conv_dst;
    const void *diff_src;
    const void *next_scale_shift;
    const void *diff_gamma;
    const void *diff_beta;
    float bwd_chan_size;

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t output_stride; // used in backward_weights only

    size_t reduce_pos_flag;
    size_t last_flag;
    size_t base_coff;
    size_t coff_thr;
};

/* pooling */
struct jit_pool_conf_t {
    int mb, c;
    int ih, iw, oh, ow;
    int stride_h, stride_w;
    int kh, kw;
    int t_pad, l_pad;
    alg_kind_t alg;
    bool is_training;
    bool pad_w_is_null;
    bool is_backward;
    data_type_t ind_dt;

    int c_block, c_tail, nb_c;
    int ur_c, ur_c_tail;
    int ur_w;
    int ur_w_tail;
    size_t tail[4];
    data_type_t src_dt;
    data_type_t dst_dt;
};

struct jit_pool_call_s {
    const float *src;
    const float *dst;
    const void *indices;
    const float *src_prf;
    const float *dst_prf;
    const void *indices_prf;
    size_t oh;
    size_t kh_padding;
    size_t kh_padding_shift;
    size_t kw_padding;
    const float* init_value;
    float ker_area_h;
};


}
}
}

#endif
