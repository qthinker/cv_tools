#pragma once
#include <cstddef>
#include <vector>
#include <opencv2\opencv.hpp>
#include <tiny_cnn/tiny_cnn.h>
#include "feature.h"
using namespace tiny_cnn;

void trans_d_smokes(const std::vector<mat3_t> & images, std::vector<vec_t> * x);
void load_frames_from_video(cv::VideoCapture & v, std::vector<cv::Mat> * images);
void load_d_from_video(const char * video_path, std::vector<std::vector<cv::Mat>> *images);
vec_t diff_two_smokes(const cv::Mat &v1, const cv::Mat &v2);
void split_d_smokes2(const std::vector<cv::Mat> & images, std::vector<mat3_t> * x, std::vector<label_t> * y, const label_t label, const int length, const int step);
void split_d_smokes(const std::vector<cv::Mat> & images, std::vector<mat3_t> * x, std::vector<label_t> * y, const label_t label, const int length, const int step);
void trans_d_smokes_diff(const std::vector<mat3_t> & images, std::vector<vec_t> * x);
void trans_d_smokes_3d(const std::vector<mat3_t> & images, std::vector<vec_t> * x);
vec_t trans_d24_smokes(const mat3_t & images);

void load_image_blocks_from_path(const char * path, std::vector<cv::Mat> *images, std::vector<int> *labels, const int label);
void load_image_blocks_from_path(const char * path, std::vector<cv::Mat> *images);
void load_image_blocks_from_path2(const char * path, std::vector<vec_t> *images, std::vector<int> *labels, const int label);
void load_image_blocks_from_path2(const char * path, std::vector<vec_t> *images);
void split_d_smokes2(const std::vector<cv::Mat> & images, std::vector<mat3_t> * x, const int length, const int step);


vec_t  image2vec(const cv::Mat img, const double scale_min = -1.0, const double scale_max = 1.0);

template<typename T>
void train_test_split(const std::vector<T> & images, const std::vector<label_t> & labels, 
					  std::vector<T> * train_images, std::vector<label_t> * train_labels,
					  std::vector<T> * test_images, std::vector<label_t> * test_labels, const double test_size)
{
	assert(images.size() == labels.size());
	int length = labels.size();
	int test_length = length * test_size;
	int train_length = length - test_length;
	train_images->resize(train_length);
	train_labels->resize(train_length);
	test_images->resize(test_length);
	test_labels->resize(test_length);

	auto iti = images.begin();
	auto itl = labels.begin();
	std::copy(iti, iti+train_length, train_images->begin());
	std::copy(iti+train_length, images.end(), test_images->begin());
	std::copy(itl, itl+train_length, train_labels->begin());
	std::copy(itl+train_length, labels.end(), test_labels->begin());
}
template<typename T>
void rand_order(std::vector<T> & x, std::vector<int> &y)
{
	assert(x.size() == y.size());
	int r1,r2;
	int length = y.size();
	for(int i = 0; i < length * 2; i++)
	{
		r1 = uniform_rand(0, length-1);
		r2 = uniform_rand(0, length-1);
		if(r1 == r2)
			continue;
		auto tmp1 = x[r1];
		x[r1] = x[r2];
		x[r2] = tmp1;
		auto tmp2 = y[r1];
		y[r1] = y[r2];
		y[r2] = tmp2;
	}
}

template<typename T>
void rand_order(std::vector<T> & x)
{
	int r1,r2;
	int length = x.size();
	for(int i = 0; i < length * 2; i++)
	{
		r1 = uniform_rand(0, length-1);
		r2 = uniform_rand(0, length-1);
		if(r1 == r2)
			continue;
		auto tmp1 = x[r1];
		x[r1] = x[r2];
		x[r2] = tmp1;
	}
}

template<typename T>
void rand_order2(std::vector<T> & x, std::vector<int> &y)
{
	assert(x.size() == y.size());
	std::vector<T> xptmp, xntmp, xtmp;
	std::vector<int> yptmp, yntmp, ytmp;

	for(int i = 0; i < y.size(); i++)
	{
		if(y[i] == 0)
		{
			xntmp.push_back(x[i]);
			yntmp.push_back(y[i]);
		}
		else
		{
			xptmp.push_back(x[i]);
			yptmp.push_back(y[i]);
		}
	}
	int pos_l = xptmp.size();
	int neg_l = xntmp.size();
	int min_l = pos_l > neg_l ? neg_l : pos_l;
	int max_l = pos_l > neg_l ? pos_l : neg_l;
	int step = 2;
	if(min_l != max_l)
		step = min_l / (max_l - min_l) + 1;
	int k = 0;
	int l = 0;
	int i = 0;
	int xstep = 0;
	while(k < pos_l && l < neg_l && i < y.size())
	{
		x[i] = xptmp[k];
		y[i] = yptmp[k];
		i++;
		k++;

		x[i] = xntmp[l];
		y[i] = yntmp[l];
		i++;
		l++;
		if(xstep % step == 0)
		{
			if(pos_l == max_l && k < pos_l && i < y.size())
			{
				x[i] = xptmp[k];
				y[i] = yptmp[k];
				i++;
				k++;
			}
			else if(neg_l == max_l && l < neg_l && i < y.size())
			{
				x[i] = xntmp[l];
				y[i] = yntmp[l];
				i++;
				l++;
			}
		}
		xstep++;
	}
	while(k < pos_l)
	{
		x[i] = xptmp[k];
		y[i] = yptmp[k];
		i++;
		k++;
	}
	while(l < neg_l)
	{
		x[i] = xntmp[l];
		y[i] = yntmp[l];
		i++;
		l++;
	}
}

//随机划分数据集
template<typename T>
void rand_split_train_test(std::vector<T> & pos_images, std::vector<T> & neg_images, 
						   std::vector<T> * train_images, std::vector<label_t> * train_labels, 
						   std::vector<T> * test_images, std::vector<label_t> * test_labels, double test_size, label_t pos_label = 1, label_t neg_label = 0)
{
	const int pos_length = pos_images.size();
	const int neg_length = neg_images.size();
	const int test_pos_length = pos_length * test_size;
	const int train_pos_length = pos_length - test_pos_length;
	const int test_neg_length = neg_length * test_size;
	const int train_neg_length = neg_length - test_neg_length;
	const int train_length = train_pos_length + train_neg_length;
	const int test_length = test_pos_length + test_neg_length;

	//
	train_images->resize(train_length);
	test_images->resize(test_length);
	train_labels->resize(train_length, neg_label);
	test_labels->resize(test_length, neg_label);

	//rand data
	rand_order(pos_images);
	rand_order(neg_images);

	//copy
	auto pos_it = pos_images.begin();
	auto neg_it = neg_images.begin();
	std::copy(pos_it, pos_it + train_pos_length, train_images->begin());
	std::copy(pos_it + train_pos_length, pos_images.end(), test_images->begin());
	std::copy(neg_it, neg_it + train_neg_length, train_images->begin() + train_pos_length);
	std::copy(neg_it + train_neg_length, neg_images.end(), test_images->begin() + test_pos_length);
	//
	for(int i = 0; i < train_pos_length; i++)
	{
		(*train_labels)[i] = pos_label;
	}
	for(int i = 0; i < test_pos_length; i++)
	{
		(*test_labels)[i] = pos_label;
	}
	//rand data
	rand_order(*train_images, *train_labels);
	rand_order(*test_images, *test_labels);
}

template<typename NN>
void extract_video_feature(NN & nn, std::vector<cv::Mat> & images, std::vector<vec_t> &x, std::vector<label_t> & y, label_t label, const int step)
{
	std::deque<vec_t> que;

	for(auto & img : images)
	{
		vec_t y_img;
		nn.predict(image2vec(img), &y_img);
		que.push_back(y_img);
		if(que.size() == 24)
		{
			vec_t x_one;
			x_one.resize(24*24);
			auto it = x_one.begin();
			for(auto & q_one : que)
			{
				std::copy(q_one.begin(), q_one.end(), it);
				it += 24;
			}
			for(int i = 0; i < step; i++)
				que.pop_front();
			x.push_back(x_one);
			y.push_back(label);
		}
	}
}
