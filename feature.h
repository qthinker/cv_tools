/*
author : wangzijie
date : 2014/11
description:
	texture features extraction under HEP(Histograms of Equivalent Patterns) .
	every img is a image block ,i use them to smoke detection.
*/

#pragma once
#include<cstddef>
#include "opencv2/opencv.hpp"
typedef std::vector<double> feature_t;
typedef std::vector<cv::Mat> mat3_t;
typedef feature_t (*p_static_feature_function)(const cv::Mat&);
typedef feature_t (*p_dynamic_feature_function)(const std::vector<cv::Mat>&, const int, const int);

feature_t get_lbp_gray(const cv::Mat & img);
feature_t get_u_lbp_gray(const cv::Mat & img);
feature_t get_ri_lbp_gray(const cv::Mat & img);
feature_t get_riu_lbp_gray(const cv::Mat & img);
feature_t get_u_lbpv_gray(const cv::Mat & img);
feature_t get_ri_lbpv_gray(const cv::Mat & img);
feature_t get_riu_lbpv_gray(const cv::Mat & img);
feature_t get_u_lbp_color(const cv::Mat & img);
feature_t get_bgc1_gray(const cv::Mat & img);
feature_t get_bgc2_gray(const cv::Mat & img);
feature_t get_bgc3_gray(const cv::Mat & img);
feature_t get_gld_gray(const cv::Mat &img);
feature_t get_sts_gray(const cv::Mat &img, const int delta);
feature_t get_rt_gray(const cv::Mat &img);
feature_t get_rtu_gray(const cv::Mat &img);
feature_t get_ilbp_gray(const cv::Mat &img);
feature_t get_3dlbp_gray(const cv::Mat &img);
feature_t get_cslbp_gray(const cv::Mat &img, const int delta);
feature_t get_mbp_gray(const cv::Mat &img);
feature_t get_dlbp_gray(const cv::Mat &img);
feature_t get_idlbp_gray(const cv::Mat &img);
feature_t get_cbp_gray(const cv::Mat &img, const int delta);
feature_t get_glbp_gray(const cv::Mat &img);
feature_t get_csts_gray(const cv::Mat &img, const int delta);
feature_t get_mts_gray(const cv::Mat &img);
feature_t get_ltp_gray(const cv::Mat &img, const int delta);
feature_t get_iltp_gray(const cv::Mat &img, const int delta);
feature_t get_eoh_gray(const cv::Mat &img);
feature_t get_eoh_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length);
feature_t get_lbp_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length);
feature_t get_bgc3_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length);
feature_t get_rtu_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length);

//2015/1/20
template<typename F>
std::vector<feature_t> get_blocks_dynamic_features(const std::deque<cv::Mat> &frames, const std::vector<cv::Rect> &rects, F dynamic_feature_function);

template<typename F>
std::vector<feature_t> get_blocks_dynamic_features(const std::deque<cv::Mat> &frames, const std::vector<cv::Point> &points, const int WD, F dynamic_feature_function);

//2014/12/6 add by wangzijie
template<typename F>
std::vector<feature_t> get_dynamic_features(const std::vector<mat3_t> &images, F dynamic_feature_function);

template<typename F>
feature_t get_top(const std::vector<cv::Mat> &frames, const int start_idx, const int border_length, F func, const int dim);
