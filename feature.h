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

feature_t get_lbp_gray(cv::Mat & img);
feature_t get_u_lbp_gray(cv::Mat & img);
feature_t get_u_lbp_color(cv::Mat & img);
feature_t get_bgc1_gray(cv::Mat & img);
feature_t get_bgc2_gray(cv::Mat & img);
feature_t get_bgc3_gray(cv::Mat & img);
feature_t get_gld_gray(cv::Mat &img);
feature_t get_sts_gray(cv::Mat &img, int delta);
feature_t get_rt_gray(cv::Mat &img);
feature_t get_rtu_gray(cv::Mat &img);
feature_t get_ilbp_gray(cv::Mat &img);
feature_t get_3dlbp_gray(cv::Mat &img);
feature_t get_cslbp_gray(cv::Mat &img, int delta);
feature_t get_mbp_gray(cv::Mat &img);
feature_t get_dlbp_gray(cv::Mat &img);
feature_t get_idlbp_gray(cv::Mat &img);
feature_t get_cbp_gray(cv::Mat &img, int delta);
feature_t get_glbp_gray(cv::Mat &img);
feature_t get_csts_gray(cv::Mat &img, int delta);
feature_t get_mts_gray(cv::Mat &img);
feature_t get_ltp_gray(cv::Mat &img, int delta);
feature_t get_iltp_gray(cv::Mat &img, int delta);
feature_t get_eoh_gray(cv::Mat &img);
feature_t get_eoh_top(std::vector<cv::Mat> frames, int start_idx, int border_length);
std::vector<feature_t> get_blocks_dynamic_features(std::deque<cv::Mat> frames, std::vector<cv::Point> points, const int WD);

