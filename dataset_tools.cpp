#include <cstddef>
#include <vector>
#include <direct.h>
#include <io.h>
#include <time.h>
#include <fstream>
#include <opencv2\opencv.hpp>
//#include "feature.h"


void load_image_blocks_from_path(const char * positive_path, const char * negative_path,
								 std::vector<cv::Mat> &images, std::vector<int> &labels,
								 int pos_label, int neg_label)
{
	cv::Mat img;
	_finddata_t fileinfo;
	int hFile;

	char old_path[_MAX_PATH];
	_getcwd(old_path,_MAX_PATH);
	_chdir(positive_path);
	hFile = _findfirst("*.jpg", &fileinfo);
	if(-1 == hFile)
	{
		std::cout<<"findfirst error!"<<std::endl;
		return ;
	}
	do{
		/* 先填正样本 */
		img = cv::imread(fileinfo.name,0);
		if(img.data == NULL)
			continue;
		images.push_back(img);
		labels.push_back(pos_label);
	}while(!_findnext(hFile, &fileinfo));
	_findclose(hFile);
	/////////////////////
	_chdir(negative_path);
	hFile = _findfirst("*.jpg", &fileinfo);
	if(-1 == hFile)
	{
		std::cout<<"findfirst error!"<<std::endl;
		return ;
	}
	do{
		/* 负样本 */
		img = cv::imread(fileinfo.name, 0);
		if(img.data == NULL)
			continue;
		images.push_back(img);
		labels.push_back(neg_label);
	}while(!_findnext(hFile, &fileinfo));
	_findclose(hFile);
	_chdir(old_path);

	/*
	打乱顺序
	*/
	srand(clock());
	int r1,r2;
	int length = labels.size();
	for(int i = 0; i < length; i++)
	{
		r1 = rand() % length;
		r2 = rand() % length;
		auto tmp1 = images[r1];
		images[r1] = images[r2];
		images[r2] = tmp1;
		auto tmp2 = labels[r1];
		labels[r1] = labels[r2];
		labels[r2] = tmp2;
	}
}

