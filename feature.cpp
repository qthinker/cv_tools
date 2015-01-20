#include "feature.h"

// uniform lbp
feature_t get_u_lbp_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbp;
	array_lbp.resize(59, 0);
	double norm = 0;
	int map_table[] = {0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,
                 15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,
                 58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
                 58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,24,58,58,
                 58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,
                 33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,58,58,58,58,58,58,58,
                 58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,
                 39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,
				 44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57};
	int lbp_value;
	int cur_value;
	int idx_;
	for(int i = 1; i < rows - 1; i++)
	{
		for(int j = 1; j < cols - 1; j++)
		{
			lbp_value = 0;
			cur_value = img.at<uchar>(i, j);
			if(img.at<uchar>(i-1, j-1) >= cur_value)
				lbp_value += 128;
			if(img.at<uchar>(i-1, j) >= cur_value)
				lbp_value += 64;
			if(img.at<uchar>(i-1, j+1) >= cur_value)
				lbp_value += 32;
			if(img.at<uchar>(i, j+1) >= cur_value)
				lbp_value += 16;
			if(img.at<uchar>(i+1, j+1) >= cur_value)
				lbp_value += 8;
			if(img.at<uchar>(i+1, j) >= cur_value)
				lbp_value += 4;
			if(img.at<uchar>(i+1, j-1) >= cur_value)
				lbp_value += 2;
			if(img.at<uchar>(i, j-1) >= cur_value)
				lbp_value += 1;
			idx_ = map_table[lbp_value];
			array_lbp[idx_]++;
		}
	}
	
	for(auto a : array_lbp)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbp.begin(); it < array_lbp.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	
	return array_lbp;
}

//rotation invariant uniform lbp
feature_t get_riu_lbp_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbp;
	array_lbp.resize(10, 0);
	double norm = 0;
	int map_table[] = {0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 1, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9,
		3, 9, 4, 5, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 
		9, 5, 6, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 5, 9, 6,
		7, 1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 2,
		3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 3, 4, 
		9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7, 4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8};
	int lbp_value;
	int cur_value;
	int idx_;
	for(int i = 1; i < rows - 1; i++)
	{
		for(int j = 1; j < cols - 1; j++)
		{
			lbp_value = 0;
			cur_value = img.at<uchar>(i, j);
			if(img.at<uchar>(i-1, j-1) >= cur_value)
				lbp_value += 128;
			if(img.at<uchar>(i-1, j) >= cur_value)
				lbp_value += 64;
			if(img.at<uchar>(i-1, j+1) >= cur_value)
				lbp_value += 32;
			if(img.at<uchar>(i, j+1) >= cur_value)
				lbp_value += 16;
			if(img.at<uchar>(i+1, j+1) >= cur_value)
				lbp_value += 8;
			if(img.at<uchar>(i+1, j) >= cur_value)
				lbp_value += 4;
			if(img.at<uchar>(i+1, j-1) >= cur_value)
				lbp_value += 2;
			if(img.at<uchar>(i, j-1) >= cur_value)
				lbp_value += 1;
			idx_ = map_table[lbp_value];
			array_lbp[idx_]++;
		}
	}
	
	for(auto a : array_lbp)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbp.begin(); it < array_lbp.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	
	return array_lbp;
}

//rotation invariant  lbp
feature_t get_ri_lbp_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbp;
	array_lbp.resize(36, 0);
	double norm = 0;
	int map_table[] = {0, 1, 1, 2, 1, 3, 2, 4, 1, 5, 3, 6, 2, 7, 4, 8, 1, 9, 5, 10, 3, 11, 6, 12, 2, 13, 7,
		14, 4, 15, 8, 16, 1, 5, 9, 13, 5, 17, 10, 18, 3, 17, 11, 19, 6, 20, 12, 21, 2, 10, 13, 22, 7, 23, 14, 
		24, 4, 18, 15, 25, 8, 26, 16, 27, 1, 3, 5, 7, 9, 11, 13, 15, 5, 17, 17, 20, 10, 23, 18, 26, 3, 11, 17, 
		23, 11, 28, 19, 29, 6, 19, 20, 30, 12, 29, 21, 31, 2, 6, 10, 14, 13, 19, 22, 25, 7, 20, 23, 30, 14, 30, 
		24, 32, 4, 12, 18, 24, 15, 29, 25, 33, 8, 21, 26, 32, 16, 31, 27, 35, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
		12, 13, 14, 15, 16, 5, 13, 17, 18, 17, 19, 20, 21, 10, 22, 23, 24, 18, 25, 26, 27, 3, 7, 11, 15, 17, 20, 
		23, 26, 11, 23, 28, 29, 19, 30, 29, 31, 6, 14, 19, 25, 20, 30, 30, 32, 12, 24, 29, 33, 21, 32, 31, 35, 2,
		4, 6, 8, 10, 12, 14, 16, 13, 18, 19, 21, 22, 24, 25, 27, 7, 15, 20, 26, 23, 29, 30, 31, 14, 25, 30, 32, 24, 
		33, 32, 35, 4, 8, 12, 16, 18, 21, 24, 27, 15, 26, 29, 31, 25, 32, 33, 35, 8, 16, 21, 27, 26, 31, 32, 35, 16, 
		27, 31, 35, 27, 35, 35, 34};
	int lbp_value;
	int cur_value;
	int idx_;
	for(int i = 1; i < rows - 1; i++)
	{
		for(int j = 1; j < cols - 1; j++)
		{
			lbp_value = 0;
			cur_value = img.at<uchar>(i, j);
			if(img.at<uchar>(i-1, j-1) >= cur_value)
				lbp_value += 128;
			if(img.at<uchar>(i-1, j) >= cur_value)
				lbp_value += 64;
			if(img.at<uchar>(i-1, j+1) >= cur_value)
				lbp_value += 32;
			if(img.at<uchar>(i, j+1) >= cur_value)
				lbp_value += 16;
			if(img.at<uchar>(i+1, j+1) >= cur_value)
				lbp_value += 8;
			if(img.at<uchar>(i+1, j) >= cur_value)
				lbp_value += 4;
			if(img.at<uchar>(i+1, j-1) >= cur_value)
				lbp_value += 2;
			if(img.at<uchar>(i, j-1) >= cur_value)
				lbp_value += 1;
			idx_ = map_table[lbp_value];
			array_lbp[idx_]++;
		}
	}
	
	for(auto a : array_lbp)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbp.begin(); it < array_lbp.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	
	return array_lbp;
}

feature_t get_u_lbp_color(const cv::Mat & img)
{
	assert(img.channels() == 3);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbp;
	array_lbp.resize(177, 0);
	cv::Mat bgr[3];
	double norm = 0;

	int map_table[] = {0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,
                 15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,
                 58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
                 58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,24,58,58,
                 58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,
                 33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,58,58,58,58,58,58,58,
                 58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,
                 39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,
				 44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57};
	int lbp_value;
	int cur_value;
	int idx_;
	cv::split(img, bgr);
	for(int k = 0; k < 3; k++)
	{
		for(int i = 1; i < rows - 1; i++)
		{
			for(int j = 1; j < cols - 1; j++)
			{
				lbp_value = 0;
				cur_value = bgr[k].at<uchar>(i, j);
				if(bgr[k].at<uchar>(i-1, j-1) >= cur_value)
					lbp_value += 128;
				if(bgr[k].at<uchar>(i-1, j) >= cur_value)
					lbp_value += 64;
				if(bgr[k].at<uchar>(i-1, j+1) >= cur_value)
					lbp_value += 32;
				if(bgr[k].at<uchar>(i, j+1) >= cur_value)
					lbp_value += 16;
				if(bgr[k].at<uchar>(i+1, j+1) >= cur_value)
					lbp_value += 8;
				if(bgr[k].at<uchar>(i+1, j) >= cur_value)
					lbp_value += 4;
				if(bgr[k].at<uchar>(i+1, j-1) >= cur_value)
					lbp_value += 2;
				if(bgr[k].at<uchar>(i, j-1) >= cur_value)
					lbp_value += 1;
				idx_ = map_table[lbp_value];
				array_lbp[k*59+idx_]++;
			}
		}
	}
	for(auto a : array_lbp)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbp.begin(); it < array_lbp.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return array_lbp;
}

//lbp
feature_t get_lbp_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t result;
	result.resize(256, 0);
	int value, center;
	double norm = 0;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			center = pi[j];
			for(int k = 0; k < 7; k++)
			{
				if(tmp[k] > center)
					value += (1 << k);
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//lbpv
feature_t get_u_lbpv_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbpv;
	array_lbpv.resize(59, 0);
	double norm = 0;
	int map_table[] = {0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,
                 15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,
                 58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
                 58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,24,58,58,
                 58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,
                 33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,58,58,58,58,58,58,58,
                 58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,
                 39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,
				 44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57};
	int lbp_value;
	int center_value;
	int idx_;
	int tmp[8];
	int mean_value;
	int var_value;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			mean_value = 0;
			var_value = 0;
			lbp_value = 0;
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			center_value = pi[j];
			for(int k = 0; k < 8; k++)
			{
				if(tmp[k] >= center_value)
					lbp_value += (1 << k);
			}
			idx_ = map_table[lbp_value];
			for(auto gp : tmp)
				mean_value += gp;
			mean_value /= 8;
			for(auto gp : tmp)
				var_value += ((gp - mean_value) * (gp - mean_value));
			var_value /= 8;
			array_lbpv[idx_] += var_value;
		}
	}
	
	for(auto a : array_lbpv)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbpv.begin(); it < array_lbpv.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	
	return array_lbpv;
}

feature_t get_ri_lbpv_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbpv;
	array_lbpv.resize(36, 0);
	double norm = 0;
	int map_table[] = {0, 1, 1, 2, 1, 3, 2, 4, 1, 5, 3, 6, 2, 7, 4, 8, 1, 9, 5, 10, 3, 11, 6, 12, 2, 13, 7,
		14, 4, 15, 8, 16, 1, 5, 9, 13, 5, 17, 10, 18, 3, 17, 11, 19, 6, 20, 12, 21, 2, 10, 13, 22, 7, 23, 14, 
		24, 4, 18, 15, 25, 8, 26, 16, 27, 1, 3, 5, 7, 9, 11, 13, 15, 5, 17, 17, 20, 10, 23, 18, 26, 3, 11, 17, 
		23, 11, 28, 19, 29, 6, 19, 20, 30, 12, 29, 21, 31, 2, 6, 10, 14, 13, 19, 22, 25, 7, 20, 23, 30, 14, 30, 
		24, 32, 4, 12, 18, 24, 15, 29, 25, 33, 8, 21, 26, 32, 16, 31, 27, 35, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
		12, 13, 14, 15, 16, 5, 13, 17, 18, 17, 19, 20, 21, 10, 22, 23, 24, 18, 25, 26, 27, 3, 7, 11, 15, 17, 20, 
		23, 26, 11, 23, 28, 29, 19, 30, 29, 31, 6, 14, 19, 25, 20, 30, 30, 32, 12, 24, 29, 33, 21, 32, 31, 35, 2,
		4, 6, 8, 10, 12, 14, 16, 13, 18, 19, 21, 22, 24, 25, 27, 7, 15, 20, 26, 23, 29, 30, 31, 14, 25, 30, 32, 24, 
		33, 32, 35, 4, 8, 12, 16, 18, 21, 24, 27, 15, 26, 29, 31, 25, 32, 33, 35, 8, 16, 21, 27, 26, 31, 32, 35, 16, 
		27, 31, 35, 27, 35, 35, 34};
	int lbp_value;
	int center_value;
	int idx_;
	int tmp[8];
	int mean_value;
	int var_value;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			mean_value = 0;
			var_value = 0;
			lbp_value = 0;
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			center_value = pi[j];
			for(int k = 0; k < 8; k++)
			{
				if(tmp[k] >= center_value)
					lbp_value += (1 << k);
			}
			idx_ = map_table[lbp_value];
			for(auto gp : tmp)
				mean_value += gp;
			mean_value /= 8;
			for(auto gp : tmp)
				var_value += ((gp - mean_value) * (gp - mean_value));
			var_value /= 8;
			array_lbpv[idx_] += var_value;
		}
	}
	
	for(auto a : array_lbpv)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbpv.begin(); it < array_lbpv.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	
	return array_lbpv;
}

feature_t get_riu_lbpv_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t array_lbpv;
	array_lbpv.resize(10, 0);
	double norm = 0;
	int map_table[] = {0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4, 1, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9,
		3, 9, 4, 5, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 
		9, 5, 6, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 5, 9, 6,
		7, 1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 2,
		3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 3, 4, 
		9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7, 4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8};
	int lbp_value;
	int center_value;
	int idx_;
	int tmp[8];
	int mean_value;
	int var_value;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			mean_value = 0;
			var_value = 0;
			lbp_value = 0;
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			center_value = pi[j];
			for(int k = 0; k < 8; k++)
			{
				if(tmp[k] >= center_value)
					lbp_value += (1 << k);
			}
			idx_ = map_table[lbp_value];
			for(auto gp : tmp)
				mean_value += gp;
			mean_value /= 8;
			for(auto gp : tmp)
				var_value += ((gp - mean_value) * (gp - mean_value));
			var_value /= 8;
			array_lbpv[idx_] += var_value;
		}
	}
	
	for(auto a : array_lbpv)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = array_lbpv.begin(); it < array_lbpv.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	
	return array_lbpv;
}

//bgc1
feature_t get_bgc1_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t result;
	result.resize(255, 0);
	int value = 0;
	double norm = 0;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			for(int k = 0; k < 7; k++)
			{
				if(tmp[k] >= tmp[k+1])
					value += (1 << k);
			}
			if(tmp[7] >= tmp[0])
				value += (1 << 7);
			result[--value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//bgc2
feature_t get_bgc2_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t result;
	result.resize(225, 0);
	int value = 0;
	double norm = 0;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			for(int k = 0; k < 4; k++)
			{
				if(tmp[2*k] >= tmp[2*(k+1)%8])
					value += (15 * (1 << k));
				if(tmp[2*k+1] >= tmp[(2*k+3)%8])
					value += (1 << k);
			}
			value -= 16;
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//bgc3
feature_t get_bgc3_gray(const cv::Mat & img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	feature_t result;
	result.resize(255, 0);
	int value;
	double norm = 0;
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			for(int k = 0; k < 8; k++)
			{
				if(tmp[3*k%8] >= tmp[3*(k+1)%8])
					value += (1 << k);
			}
			result[--value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}


//EOH features
feature_t get_eoh_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int Orientation[8] = {0};
	int Magnitudes[8] = {0};
	cv::Mat H, V, M;

	cv::Sobel(img, H, CV_32F, 1, 0);
	cv::Sobel(img, V, CV_32F, 0, 1);
	cv::sqrt(H.mul(H)+V.mul(V), M);

	float *ph, *pv, *pm;
	float h_ij, v_ij;
	int Ori, Midx;
	for(int i = 0; i < rows; i++)
	{
		ph = H.ptr<float>(i);
		pv = V.ptr<float>(i);
		pm = M.ptr<float>(i);
		for(int j = 0; j < cols; j++)
		{
			h_ij = ph[j];
			v_ij = pv[j];
			if(h_ij > 0 && v_ij > 0)
				Ori = std::atan(v_ij / h_ij) * 180 / CV_PI;
			else if(h_ij < 0 && v_ij != 0)
				Ori = 180 + std::atan(v_ij / h_ij) * 180 / CV_PI;
			else if(h_ij > 0 && v_ij < 0)
				Ori = 360 + std::atan(v_ij / h_ij) * 180 / CV_PI;
			else 
				Ori = 1;
			Orientation[Ori / 45] += 1;
			Midx = std::min(int(pm[j]/12.5), 7);
			Magnitudes[Midx] += 1;
		}
	}

	double o_sum = 0;
	double m_sum = 0;
	feature_t result;
	for(int k = 0; k < 8; k++)
	{
		o_sum += Orientation[k];
		m_sum += Magnitudes[k];
	}
	for(int k = 0; k < 8; k++)
	{
		result.push_back(Orientation[k]/o_sum);
	}
	for(int k = 0; k < 8; k++)
	{
		result.push_back(Magnitudes[k]/m_sum);
	}

	return result;
}

//2014/11/3 add
//Gray Level Differences£¬GLD
feature_t get_gld_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, i4, i5, i6, i7;
	int value;
	double norm = 0;
	feature_t result;
	result.resize(256, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			center = pi[j];
			i4 = pi[j+1];
			i5 = pi_[j+1];
			i6 = pi_[j];
			i7 = pi_[j-1];
			value = (std::abs(i4-center) + std::abs(i5-center) + std::abs(i6-center) + std::abs(i7-center)) / 4;
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Simplified Texture Spectrum£¬STS
feature_t get_sts_gray(const cv::Mat &img, int delta)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, a[4], s;
	int value;
	double norm = 0;
	feature_t result;
	result.resize(81, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			center = pi[j];
			a[0] = pi[j+1];
			a[1] = pi_[j+1];
			a[2] = pi_[j];
			a[3] = pi_[j-1];
			s = 1;
			value = 0;
			for(int k = 0; k < 4; k++)
			{
				int diff = center - a[i];
				if(diff > delta)
					value += 2 * s;
				else if(diff < -delta)
					;
				else
					value += s;
				s *= 3;
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Rank Transform£¬RT
feature_t get_rt_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, value;
	double norm = 0;
	feature_t result;
	result.resize(9, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			center = pi[j];
			if(pi_[j-1] < center)
				value++;
			if(pi_[j] < center)
				value++;
			if(pi_[j+1] < center)
				value++;
			if(pi[j-1] < center)
				value++;
			if(pi[j+1] < center)
				value++;
			if(pi_p[j-1] < center)
				value++;
			if(pi_p[j] < center)
				value++;
			if(pi_p[j+1] < center)
				value++;
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Reduced Textue Uint,RTU
feature_t get_rtu_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, value;
	double norm = 0;
	feature_t result;
	result.resize(45, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			center = pi[j];
			int a0 = 0;
			int a1 = 0;
			if(pi_[j-1] < center)
				a0++;
			else if(pi_[j-1] == center)
				a1++;
			if(pi_[j] < center)
				a0++;
			else if(pi_[j] == center)
				a1++;
			if(pi_[j+1] < center)
				a0++;
			else if(pi_[j+1] == center)
				a1++;
			if(pi[j-1] < center)
				a0++;
			else if(pi[j-1] == center)
				a1++;
			if(pi[j+1] < center)
				a0++;
			else if(pi[j+1] == center)
				a1++;
			if(pi_p[j-1] < center)
				a0++;
			else if(pi_p[j-1] == center)
				a1++;
			if(pi_p[j] < center)
				a0++;
			else if(pi_p[j] == center)
				a1++;
			if(pi_p[j+1] < center)
				a0++;
			else if(pi_p[j+1] == center)
				a1++;
			value = a0 + (8 - a1) * (9 - a1) / 2;
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Improved Local Binary Patterns, ILBP
feature_t get_ilbp_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int mean, value;
	double norm = 0;
	feature_t result;
	result.resize(511, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			mean = 0;
			int tmp[9];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			tmp[8] = pi[j];
			for(int k = 0; k < 9; k++)
				mean += tmp[k];
			mean /= 9;
			for(int k = 0; k < 9; k++)
			{
				if(tmp[k] >= mean)
					value += (1 << k);
			}
			result[--value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//3D Local Binary Patterns,3DLBP
feature_t get_3dlbp_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, value;
	double norm = 0;
	feature_t result;
	result.resize(1024, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			center = pi[j];
			for(int k = 0; k < 8; k++)
			{
				if(tmp[k] > center)
					value += (1 << k);
			}
			result[value]++;
			for(int l = 0; l < 3; l++)
			{
				value = 0;
				for(int k = 0; k < 8; k++)
				{
					if((std::abs(tmp[k] - center)) & (1 << l))
						value += (1 << k);
				}
				result[value+256*(1+l)]++;
			}
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Center-symmetric Local Binary Patterns,CS-LBP
feature_t get_cslbp_gray(const cv::Mat &img, int delta)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int value;
	double norm = 0;
	feature_t result;
	result.resize(16, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			for(int k = 0; k < 4; k++)
			{
				if(tmp[k] - tmp[k+4] - delta > 1)
					value += (1 << k);
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Median Binary Patterns,MBP
feature_t get_mbp_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int mid, value;
	double norm = 0;
	feature_t result;
	result.resize(511, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[9];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			tmp[8] = pi[j];
			std::sort(&tmp[0], &tmp[8]);
			mid = tmp[4];
			for(int k = 0; k < 9; k++)
			{
				if(tmp[k] >= mid)
					value += (1 << k);
			}
			result[--value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Improved Centre-Symmetric Local Binary Patterns,D-LBP
feature_t get_dlbp_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, value;
	double norm = 0;
	feature_t result;
	result.resize(16, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			center = pi[j];
			for(int k = 0; k < 4; k++)
			{
				if((tmp[k] >= center && center >= tmp[k+4]) || (tmp[k] < center && center < tmp[k+4]))
					value += (1 << k);
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Improved Centre-Symmetric Local Binary Patterns
feature_t get_idlbp_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int mean, value;
	double norm = 0;
	feature_t result;
	result.resize(16, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			mean = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			for(int k = 0; k < 8; k++)
				mean += tmp[k];
			mean /= 8;
			for(int k = 0; k < 4; k++)
			{
				if((tmp[k] >= mean && tmp[k+4] >= mean) || (tmp[k] < mean && tmp[k+4] < mean))
					value += (1 << k);
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Centralized Binary Patterns,CBP
feature_t get_cbp_gray(const cv::Mat &img, int delta)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int mean, value;
	double norm = 0;
	feature_t result;
	result.resize(32, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			mean = 0;
			int tmp[9];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			for(int k = 0; k < 8; k++)
				mean += tmp[k];
			mean /= 8;
			for(int k = 0; k < 4; k++)
			{
				if(std::abs(tmp[k]-tmp[k+4]) >= delta)
					value += (1 << k);
			}
			if(std::abs(pi[j]-mean) >= delta)
				value += 16;
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Gradient-based Local Binary Patterns,GLBP
feature_t get_glbp_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int mean, center, value;
	double norm = 0;
	feature_t result;
	result.resize(256, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			mean = 0;
			center = pi[j];
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			mean = (std::abs(tmp[0] - tmp[4]) + std::abs(tmp[2] - tmp[6])) / 2;
			for(int k = 0; k < 8; k++)
			{
				if(mean >= std::abs(tmp[k] - center))
					value += (1 << k);
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//center-symmetric texture spectrum
feature_t get_csts_gray(const cv::Mat &img, int delta)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int value;
	double norm = 0;
	feature_t result;
	result.resize(81, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			int s = 1;
			for(int k = 0; k < 4; k++)
			{
				int diff = tmp[k] - tmp[k+4];
				if(diff < -delta)
					;
				else if(diff > delta)
					value += s*2;
				else
					value += s;
				s *= 3;
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//modified texure spectrum
feature_t get_mts_gray(const cv::Mat &img)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, value;
	double norm = 0;
	feature_t result;
	result.resize(16, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value = 0;
			center = pi[j];
			int tmp[4];
			tmp[0] = pi[j+1];
			tmp[1] = pi_[j+1];
			tmp[2] = pi_[j];
			tmp[3] = pi_[j-1];
			
			for(int k = 0; k < 4; k++)
			{
				if(tmp[k] >= center)
					value += (1 << k);
			}
			result[value]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//ltp
feature_t get_ltp_gray(const cv::Mat &img, int delta)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int center, value1, value2;
	double norm = 0;
	feature_t result;
	result.resize(512, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value1 = 0;
			value2 = 0;
			center = pi[j];
			int tmp[8];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			
			for(int k = 0; k < 8; k++)
			{
				if(tmp[k] - center >= delta)
					value1 += (1 << k);
				if(center - tmp[k] >= delta)
					value2 += (1 << k);
			}
			result[value1]++;
			result[256+value2]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

//Improved Local Ternary Patterns, ILTP
feature_t get_iltp_gray(const cv::Mat &img, int delta)
{
	assert(img.dims == 2);
	int rows = img.rows;
	int cols = img.cols;
	int mean, value1, value2;
	double norm = 0;
	feature_t result;
	result.resize(1024, 0);
	for(int i = 1; i < rows - 1; i++)
	{
		const uchar * pi = img.ptr<uchar>(i);
		const uchar * pi_ = pi - img.step;
		const uchar * pi_p = pi + img.step;
		for(int j = 1; j < cols - 1; j++)
		{
			value1 = 0;
			value2 = 0;
			mean = 0;
			int tmp[9];
			tmp[0] = pi[j-1];
			tmp[1] = pi_p[j-1];
			tmp[2] = pi_p[j];
			tmp[3] = pi_p[j+1];
			tmp[4] = pi[j+1];
			tmp[5] = pi_[j+1];
			tmp[6] = pi_[j];
			tmp[7] = pi_[j-1];
			tmp[8] = pi[j];
			for(int k = 0; k < 9; k++)
				mean += tmp[k];
			mean /= 9;
			for(int k = 0; k < 9; k++)
			{
				if(tmp[k] - mean >= delta)
					value1 += (1 << k);
				if(mean - tmp[k] >= delta)
					value2 += (1 << k);
			}
			result[value1]++;
			result[512+value2]++;
		}
	}
	for(auto a : result)
	{
		norm += a;
	}
	norm = std::sqrt(norm) + 1e-10;
	for(auto it = result.begin(); it < result.end(); it++)
	{
		*it = std::sqrt(*it) / norm;
	}
	return result;
}

feature_t get_eoh_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length)
{
	return get_top(frames, start_idx, border_length, get_eoh_gray, 16);
}

feature_t get_lbp_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length)
{
	return get_top(frames, start_idx, border_length, get_u_lbp_gray, 59);
}

feature_t get_bgc3_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length)
{
	return get_top(frames, start_idx, border_length, get_bgc3_gray, 255);
}

feature_t get_rtu_top(const std::vector<cv::Mat> & frames, const int start_idx, const int border_length)
{
	return get_top(frames, start_idx, border_length, get_rtu_gray, 45);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//2015/1/20
template<typename F>
std::vector<feature_t> get_blocks_dynamic_features(const std::deque<cv::Mat> &frames, const std::vector<cv::Rect> &rects, F dynamic_feature_function)
{
	assert(frames.size() != 0 && frames[0].dims == 2);
	const int num = rects.size();
	const int frame_length = frames.size();
	std::vector<feature_t> result;
	for(int i = 0; i < num; i++)
	{
		std::vector<cv::Mat> blocks;
		for(int k = 0; k < frame_length; k++)
		{
			cv::Mat block = frames[k](rects[i]);
			blocks.push_back(block.clone());
		}
		auto feature_i = dynamic_feature_function(blocks, 0, 1);
		result.push_back(feature_i);
	}
	return result;
}
template std::vector<feature_t> get_blocks_dynamic_features<p_dynamic_feature_function>(const std::deque<cv::Mat> &frames, const std::vector<cv::Rect> &rects, p_dynamic_feature_function dynamic_feature_function);

template<typename F>
std::vector<feature_t> get_blocks_dynamic_features(const std::deque<cv::Mat> &frames, const std::vector<cv::Point> &points, const int WD, F dynamic_feature_function)
{
	assert(frames.size() != 0 && frames[0].dims == 2);
	const int num = points.size();
	const int frame_length = frames.size();
	std::vector<feature_t> result;
	for(int i = 0; i < num; i++)
	{
		std::vector<cv::Mat> blocks;
		for(int k = 0; k < frame_length; k++)
		{
			cv::Mat block = frames[k](cv::Rect(points[i].x, points[i].y, WD, WD));
			blocks.push_back(block.clone());
		}
		auto feature_i = dynamic_feature_function(blocks, 0, 1);
		result.push_back(feature_i);
	}
	return result;
}
template std::vector<feature_t> get_blocks_dynamic_features<p_dynamic_feature_function>(const std::deque<cv::Mat> &frames, const std::vector<cv::Point> &points, const int WD, p_dynamic_feature_function dynamic_feature_function);

template<typename F>
feature_t get_top(const std::vector<cv::Mat> &frames, const int start_idx, const int border_length, F func, const int dim)
{
	feature_t h1, h2, h3;
	h1.resize(dim, 0.0);
	h2.resize(dim, 0.0);
	h3.resize(dim, 0.0);
	int frame_length = frames.size();
	int height = frames[0].rows;
	int width = frames[0].cols;

	for(int i = start_idx; i < frame_length; i++)
	{
		auto xy_i_plane = func(frames[i]);
		for(int k = 0; k < dim; k++)
		{
			h1[k] += xy_i_plane[k];
		}
	}

	cv::Mat xt(frame_length, width, CV_8UC1);
	for(int yc = border_length; yc < height - border_length; yc += 2)
	{
		for(int k = 0; k < frame_length; k++)
		{
			memcpy(xt.ptr<uchar>(k), frames[k].ptr<uchar>(yc), width);
		}
		auto xt_yc_plane = func(xt);
		for(int k = 0; k < dim; k++)
		{
			h2[k] += xt_yc_plane[k];
		}
	}

	cv::Mat yt(frame_length, height, CV_8UC1);
	for(int xc = border_length; xc < height - border_length; xc += 2)
	{
		for(int k = 0; k < frame_length; k++)
		{
			cv::Mat tmp;
			tmp = frames[k].t();
			memcpy(yt.ptr<uchar>(k), tmp.ptr<uchar>(xc), height);
		}
		auto yt_xc_plane = func(yt);
		for(int k = 0; k < dim; k++)
		{
			h3[k] += yt_xc_plane[k];
		}
	}
	
	double sum1, sum2, sum3;
	sum1 = 0; 
	sum2 = 0; 
	sum3 = 0;
	for(int k = 0; k < dim; k++)
	{
		sum1 += h1[k];
		sum2 += h2[k];
		sum3 += h3[k];
	}
	for(int k = 0; k < dim; k++)
	{
		h1[k] /= sum1;
		h2[k] /= sum2;
		h3[k] /= sum3;
	}
	auto result = h1;
	for(int k = 0; k < dim; k++)
		result.push_back(h2[k]);
	for(int k = 0; k < dim; k++)
		result.push_back(h3[k]);
	return result;
}
template feature_t get_top<p_static_feature_function>(const std::vector<cv::Mat> &frames, const int start_idx, const int border_length, p_static_feature_function func, const int dim);

template<typename F>
std::vector<feature_t> get_dynamic_features(const std::vector<mat3_t> &images, F dynamic_feature_function)
{
	std::vector<feature_t> result;
	for(auto & frames : images)
	{
		auto feat = dynamic_feature_function(frames, 0, 1);
		result.push_back(feat);
	}
	return result;
}
template std::vector<feature_t> get_dynamic_features<p_dynamic_feature_function>(const std::vector<mat3_t> &images, p_dynamic_feature_function dynamic_feature_function);
