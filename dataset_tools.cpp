#include <cstddef>
#include <vector>
#include <regex>
#include <direct.h>
#include <io.h>
#include <time.h>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <tiny_cnn/tiny_cnn.h>
using namespace tiny_cnn;
//#include "feature.h"
#include "dataset_tools.h"

//递归处理子文件夹, 读取images到std::vector<cv::Mat>
void load_image_blocks_from_path(const char * path, std::vector<cv::Mat> *images, std::vector<int> *labels, const int label)
{
	_finddata_t fileinfo;
	int hFile;

	char old_path[_MAX_PATH];
	_getcwd(old_path,_MAX_PATH);
	_chdir(path);
	hFile = _findfirst("*", &fileinfo);
	if(hFile != -1)
	{
		do{
			if(fileinfo.attrib == _A_SUBDIR)
			{
				if(0 != strcmp(fileinfo.name, ".") && 0 != strcmp(fileinfo.name , ".."))
					load_image_blocks_from_path(fileinfo.name, images, labels, label);
			}
			else
			{
				auto img = cv::imread(fileinfo.name,0);
				if(img.data == NULL)
					continue;
				images->push_back(img);
				labels->push_back(label);
			}
		}while(!_findnext(hFile, &fileinfo));
		_findclose(hFile);
	}
	_chdir(old_path);
}

//递归处理子文件夹, 读取images到std::vector<cv::Mat>
void load_image_blocks_from_path(const char * path, std::vector<cv::Mat> *images)
{
	_finddata_t fileinfo;
	int hFile;

	char old_path[_MAX_PATH];
	_getcwd(old_path,_MAX_PATH);
	_chdir(path);
	hFile = _findfirst("*", &fileinfo);
	if(hFile != -1)
	{
		do{
			if(fileinfo.attrib == _A_SUBDIR)
			{
				if(0 != strcmp(fileinfo.name, ".") && 0 != strcmp(fileinfo.name , ".."))
					load_image_blocks_from_path(fileinfo.name, images);
			}
			else
			{
				auto img = cv::imread(fileinfo.name,0);
				if(img.data == NULL)
					continue;
				images->push_back(img);
			}
		}while(!_findnext(hFile, &fileinfo));
		_findclose(hFile);
	}
	_chdir(old_path);
}

//cv::Mat ==>> vec_t
vec_t  image2vec(const cv::Mat img, const double scale_min, const double scale_max)
{
	vec_t X;
	int rows = img.rows;
	int cols = img.cols;
	X.resize(rows * cols);
	for(int i = 0; i < rows; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		for(int j = 0; j < cols; j++)
		{
			X[i*cols+j] = (pi[j] / 255.0) * (scale_max - scale_min) + scale_min;
		}
	}
	return X;
}

//读取images到std::vector<vec_t>
void load_image_blocks_from_path2(const char * path, std::vector<vec_t> *images, std::vector<int> *labels, const int label)
{
	_finddata_t fileinfo;
	int hFile;

	char old_path[_MAX_PATH];
	_getcwd(old_path,_MAX_PATH);
	_chdir(path);
	hFile = _findfirst("*", &fileinfo);
	if(hFile != -1)
	{
		do{
			if(fileinfo.attrib == _A_SUBDIR)
			{
				if(0 != strcmp(fileinfo.name, ".") && 0 != strcmp(fileinfo.name , ".."))
					load_image_blocks_from_path2(fileinfo.name, images, labels, label);
			}
			else
			{
				auto img = cv::imread(fileinfo.name,0);
				if(img.data == NULL)
					continue;
				images->push_back(image2vec(img));
				labels->push_back(label);
			}
		}while(!_findnext(hFile, &fileinfo));
		_findclose(hFile);
	}
	_chdir(old_path);
}

//读取images到std::vector<vec_t>
void load_image_blocks_from_path2(const char * path, std::vector<vec_t> *images)
{
	_finddata_t fileinfo;
	int hFile;

	char old_path[_MAX_PATH];
	_getcwd(old_path,_MAX_PATH);
	_chdir(path);
	hFile = _findfirst("*", &fileinfo);
	if(hFile != -1)
	{
		do{
			if(fileinfo.attrib == _A_SUBDIR)
			{
				if(0 != strcmp(fileinfo.name, ".") && 0 != strcmp(fileinfo.name , ".."))
					load_image_blocks_from_path2(fileinfo.name, images);
			}
			else
			{
				auto img = cv::imread(fileinfo.name,0);
				if(img.data == NULL)
					continue;
				images->push_back(image2vec(img));
			}
		}while(!_findnext(hFile, &fileinfo));
		_findclose(hFile);
	}
	_chdir(old_path);
}

//隔帧采样
void split_d_smokes2(const std::vector<cv::Mat> & images, std::vector<mat3_t> * x, std::vector<label_t> * y, const label_t label, const int length, const int step)
{
	std::deque<cv::Mat> que1, que2;
	int idx  = 0;
	for(auto _one : images)
	{
		if(idx % 2 == 0)
		{
			que1.push_back(_one);
			if(que1.size() == length)
			{
				mat3_t x_one;
				for(auto & img : que1)
					x_one.push_back(img.clone());
				x->push_back(x_one);
				y->push_back(label);
				for(int i = 0; i < step; i++)
					que1.pop_front();
			}
		}
		else
		{
			que2.push_back(_one);
			if(que2.size() == length)
			{
				mat3_t x_one;
				for(auto & img : que2)
					x_one.push_back(img.clone());
				x->push_back(x_one);
				y->push_back(label);
				for(int i = 0; i < step; i++)
					que2.pop_front();
			}
		}
		idx++;
	}
}

//隔帧采样
void split_d_smokes2(const std::vector<cv::Mat> & images, std::vector<mat3_t> * x, const int length, const int step)
{
	std::deque<cv::Mat> que1, que2;
	int idx  = 0;
	for(auto _one : images)
	{
		if(idx % 2 == 0)
		{
			que1.push_back(_one);
			if(que1.size() == length)
			{
				mat3_t x_one;
				for(auto & img : que1)
					x_one.push_back(img.clone());
				x->push_back(x_one);
				for(int i = 0; i < step; i++)
					que1.pop_front();
			}
		}
		else
		{
			que2.push_back(_one);
			if(que2.size() == length)
			{
				mat3_t x_one;
				for(auto & img : que2)
					x_one.push_back(img.clone());
				x->push_back(x_one);
				for(int i = 0; i < step; i++)
					que2.pop_front();
			}
		}
		idx++;
	}
}

//连续采样
void split_d_smokes(const std::vector<cv::Mat> & images, std::vector<mat3_t> * x, std::vector<label_t> * y, const label_t label, const int length, const int step)
{
	std::deque<cv::Mat> que;
	for(auto _one : images)
	{
		que.push_back(_one);
		if(que.size() == length)
		{
			mat3_t x_one;
			for(auto & img : que)
				x_one.push_back(img.clone());
			x->push_back(x_one);
			y->push_back(label);
			for(int i = 0; i < step; i++)
				que.pop_front();
		}
	}
}

//std::vector<mat3_t> =>> std::vector<vec_t>
void trans_d_smokes(const std::vector<mat3_t> & images, std::vector<vec_t> * x)
{
	assert(images.size() != 0);
	auto & m0 = images[0];
	const int depth = m0.size();
	const int width = m0[0].cols;
	const int height = m0[0].rows;
	const int x_one_length = depth * width * height;
	const int x_one_vec_dim = width * height;

	for(auto & m : images)
	{
		vec_t x_one;
		x_one.resize(x_one_length);
		int idx = 0;
		for(auto & img : m)
		{
			auto vec_img = image2vec(img);
			std::copy(vec_img.begin(), vec_img.end(), x_one.begin()+idx*x_one_vec_dim);
			idx++;
		}
		x->push_back(x_one);
	}
}

//std::vector<mat3_t> =>> std::vector<vec_t>, diff
void trans_d_smokes_diff(const std::vector<mat3_t> & images, std::vector<vec_t> * x)
{
	assert(images.size() != 0);
	auto & m0 = images[0];
	const int depth = m0.size();
	const int width = m0[0].cols;
	const int height = m0[0].rows;
	const int x_one_length = (depth-1) * width * height;
	const int x_one_vec_dim = width * height;

	for(auto & m : images)
	{
		vec_t x_one;
		x_one.resize(x_one_length);
		int idx = 0;
		for(int i = 1; i < m.size(); i++)
		{
			auto vec_img = diff_two_smokes(m[i-1], m[i]);
			std::copy(vec_img.begin(), vec_img.end(), x_one.begin()+idx*x_one_vec_dim);
			idx++;
		}
		x->push_back(x_one);
	}
}

vec_t diff_two_smokes(const cv::Mat &v1, const cv::Mat &v2)
{
	assert(v1.size() == v2.size());
	const int rows = v1.rows;
	const int cols = v1.cols;
	vec_t result;
	result.resize(cols * rows, 0);
	double t1, t2;
	for(int i = 0; i < rows; i++)
	{
		const uchar * pi1 = v1.ptr<uchar>(i);
		const uchar * pi2 = v2.ptr<uchar>(i);
		for(int j = 0; j < cols; j++)
		{
			t1 = pi1[j];
			t2 = pi2[j];
			auto dd = t1 - t2;
			dd /= 50.0;
			if(dd > 1.0)
				dd = 1.0;
			else if(dd < -1.0)
				dd = -1.0;
			result[i*cols+j] = dd;
		}
	}
	return result;
}

void load_frames_from_video(cv::VideoCapture & v, std::vector<cv::Mat> * images)
{
	cv::Mat img;
	while(v.read(img))
	{
		images->push_back(img.clone());
	}
}

//读取动态smoke
void load_d_from_video(const char * video_path, std::vector<std::vector<cv::Mat>> *images)
{
	if(!_access(video_path, 0))
	{
		_finddata_t fileinfo;
		int hFile;
		char old_path[_MAX_PATH];
		_getcwd(old_path, _MAX_PATH);
		_chdir(video_path);
		if((hFile = _findfirst("*.avi", &fileinfo)) != -1)
		{
			do{
				cv::VideoCapture v(fileinfo.name);
				std::vector<cv::Mat> imgs;
				load_frames_from_video(v, &imgs);
				images->push_back(imgs);
				v.release();
			}while(!_findnext(hFile, &fileinfo));
		}
		_chdir(old_path);
	}
}

void trans_d_smokes_3d(const std::vector<mat3_t> & images, std::vector<vec_t> * x)
{
	for(auto & v : images)
	{
		x->push_back(trans_d24_smokes(v));
	}
}

//24x24x24 mat3_t ==>> vec_t 12 x 3 共 36 channels, dim = 36 x 24 x 24
vec_t trans_d24_smokes(const mat3_t & images)
{
	assert((images.size() == 24) && (images[0].rows == 24) && (images[0].cols == 24));
	vec_t result;
	result.resize(24*24*36);
	int idx = 0;
	for(int i = 0; i < 24; i += 2)
	{
		auto vec = image2vec(images[i]);
		std::copy(vec.begin(), vec.end(), result.begin()+24*24*idx);
		idx++;
	}

	cv::Mat xt(24, 24, CV_8UC1);
	for(int yc = 0; yc < 24; yc += 2)
	{
		for(int k = 0; k < 24; k++)
		{
			memcpy(xt.ptr<uchar>(k), images[k].ptr<uchar>(yc), 24);
		}
		auto vec = image2vec(xt);
		std::copy(vec.begin(), vec.end(), result.begin()+24*24*idx);
		idx++;
	}

	cv::Mat yt(24, 24, CV_8UC1);
	for(int xc = 0; xc < 24; xc += 2)
	{
		for(int k = 0; k < 24; k++)
		{
			cv::Mat tmp;
			tmp = images[k].t();
			memcpy(yt.ptr<uchar>(k), tmp.ptr<uchar>(xc), 24);
		}
		auto vec = image2vec(xt);
		std::copy(vec.begin(), vec.end(), result.begin()+24*24*idx);
		idx++;
	}

	return result;
}