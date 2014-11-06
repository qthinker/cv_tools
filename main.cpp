#include <fstream>
#include <time.h>
#include "feature.h"
#include "classify_tools.h"
#include "dataset_tools.h"
#include <tbb/task_group.h>
#define CNN_USE_TBB
#include <tiny_cnn/tiny_cnn.h>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void svm_train_test(int iters = 10);
void mlp_train_test();

///////////////////////////////// global
p_func_feature_extract funcs[] = {get_lbp_gray, get_u_lbp_gray, get_bgc1_gray, get_bgc2_gray, get_bgc3_gray,
		get_rt_gray, get_rtu_gray, get_ilbp_gray, get_3dlbp_gray,get_mbp_gray, get_dlbp_gray, get_idlbp_gray,
		get_glbp_gray, get_mts_gray, get_eoh_gray};
const char * func_names[] = {"get_lbp_gray","get_u_lbp_gray", "get_bgc1_gray", "get_bgc2_gray", "get_bgc3_gray",
		"get_rt_gray", "get_rtu_gray", "get_ilbp_gray", "get_3dlbp_gray", "get_mbp_gray", "get_dlbp_gray", "get_idlbp_gray",
		"get_glbp_gray", "get_mts_gray", "get_eoh_gray"};
const char * func_models[] = {"lbp_gray.model","u_lbp_gray.model", "bgc1_gray.model", "bgc2_gray.model", "bgc3_gray.model",
		"rt_gray.model", "rtu_gray.model", "ilbp_gray.model", "3dlbp_gray.model", "mbp_gray.model", "dlbp_gray.model", "idlbp_gray.model",
		"glbp_gray.model", "mts_gray.model", "eoh_gray.model"};
p_func_feature_extract_with_delta funcs2[] = {get_sts_gray,get_cslbp_gray,get_cbp_gray,get_csts_gray,get_ltp_gray,get_iltp_gray};
const char * func_names2[] = {"get_sts_gray","get_cslbp_gray","get_cbp_gray","get_csts_gray","get_ltp_gray","get_iltp_gray"};
const char * func_models2[] = {"sts_gray.model","cslbp_gray.model","cbp_gray.model","csts_gray.model","ltp_gray.model","iltp_gray.model"};

int main()
{
	//auto ret = mlp_test("F:\\DataSet\\smoke\\image\\my24\\smoke\\light-smoke","F:\\DataSet\\smoke\\image\\my24\\nonsmoke", "ulbp-weights", get_u_lbp_gray);
	//ret.print_detail(std::cout);
	//mlp_train("F:\\DataSet\\smoke\\image\\my24\\smoke","F:\\DataSet\\smoke\\image\\my24\\nonsmoke","ulbp-weights",get_u_lbp_gray);
	//svm_train_test(1);
	mlp_train_test();
	std::cout<<"complete press any key out"<<std::endl;
	getchar();
}
void mlp_train_test()
{
	//load dataset
	//std::vector<cv::Mat> images;
	//std::vector<int> labels;
	std::vector<cv::Mat> train_images;
	std::vector<cv::Mat> test_images;
	std::vector<int> train_labels;
	std::vector<int> test_labels;
	boost::timer t;
	load_image_blocks_from_path("d:\\img\\24\\smoke", "d:\\img\\24\\nonsmoke", train_images, train_labels, 1, 0);
	load_image_blocks_from_path("F:\\DataSet\\smoke\\image\\my24\\smoke\\light-smoke", "F:\\DataSet\\smoke\\image\\my24\\nonsmoke", test_images, test_labels, 1, 0);
	std::cout<<"load dataset complete! use time : "<<t.elapsed()<<"s"<<std::endl;
	std::ofstream ofs("my24-mlp-yuan-result.txt");

	//split dataset
	/*
	std::vector<cv::Mat> train_images;
	std::vector<cv::Mat> test_images;
	std::vector<int> train_labels;
	std::vector<int> test_labels;
	train_test_split(images, labels, train_images, train_labels, test_images, test_labels, 0.5);
	*/
	for(auto func : funcs)
	{
		static int idx = 0;
		//feature extract
		auto train_feats = get_features_from_images(train_images, func);
		auto test_feats = get_features_from_images(test_images, func);

		//construct mlp
		const int num_input = train_feats[0].size();
		const int num_hidden_units = 30;
		int num_units[] = { num_input, num_hidden_units, 2 };
		auto nn = make_mlp<mse, gradient_descent_levenberg_marquardt, tan_h>(num_units, num_units + 3);

		//train mlp
		nn.optimizer().alpha = 0.005;
		 boost::progress_display disp(train_feats.size());
		t.restart();
		// create callback
		auto on_enumerate_epoch = [&](){
			std::cout << t.elapsed() << "s elapsed." << std::endl;
			tiny_cnn::result res = nn.test(test_feats, test_labels);
			std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;
			nn.optimizer().alpha *= 0.85; // decay learning rate
			nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);
			disp.restart(train_feats.size());
			t.restart();
		};
		auto on_enumerate_data = [&](){ 
			++disp; 
		};  

		nn.train(train_feats, train_labels, 1, 20, on_enumerate_data, on_enumerate_epoch);
		std::cout<<func_names[idx]<<std::endl;
		nn.test(test_feats, test_labels).print_detail(std::cout);
		ofs<<func_names[idx]<<std::endl;
		nn.test(test_feats, test_labels).print_detail(ofs);
		idx++;
	}
	for(auto func : funcs2)
	{
		static int idx = 0;
		//feature extract
		auto train_feats = get_features_from_images(train_images, func, 10);
		auto test_feats = get_features_from_images(test_images, func, 10);

		//construct mlp
		const int num_input = train_feats[0].size();
		const int num_hidden_units = 30;
		int num_units[] = { num_input, num_hidden_units, 2 };
		auto nn = make_mlp<mse, gradient_descent_levenberg_marquardt, tan_h>(num_units, num_units + 3);

		//train mlp
		nn.optimizer().alpha = 0.005;
		 boost::progress_display disp(train_feats.size());
		t.restart();
		// create callback
		auto on_enumerate_epoch = [&](){
			std::cout << t.elapsed() << "s elapsed." << std::endl;
			tiny_cnn::result res = nn.test(test_feats, test_labels);
			std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;
			nn.optimizer().alpha *= 0.85; // decay learning rate
			nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);
			disp.restart(train_feats.size());
			t.restart();
		};
		auto on_enumerate_data = [&](){ 
			++disp; 
		};  

		nn.train(train_feats, train_labels, 1, 30, on_enumerate_data, on_enumerate_epoch);
		std::cout<<func_names2[idx]<<std::endl;
		nn.test(test_feats, test_labels).print_detail(std::cout);
		ofs<<func_names2[idx]<<std::endl;
		nn.test(test_feats, test_labels).print_detail(ofs);
		idx++;
	}
	ofs.close();
}
void svm_train_test( int iters)
{
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	boost::timer t;
	load_image_blocks_from_path("D:\\img\\24\\smoke", "D:\\img\\24\\nonsmoke", images, labels, 1, -1);
	std::cout<<"load dataset complete! use time : "<<t.elapsed()<<"s"<<std::endl;
	std::ofstream ofs("2-feature-result.txt");
	//
	std::vector<cv::Mat> train_images, test_images;
	std::vector<int> train_labels, test_labels;
	std::vector<double> accs;
	//iter 10s
	t.restart();
	for(int i = 0; i < iters; i++)
	{
		std::cout<<"iter: "<<i<<std::endl;
		train_test_split(images, labels, train_images, train_labels, test_images, test_labels, 0.5);
		auto idx = 0;
		for(auto func : funcs)
		{
			auto train_feats = get_features_from_images(train_images, func);
			auto test_feats = get_features_from_images(test_images, func);
			auto acc = svm_train_and_test(train_feats, train_labels, test_feats, test_labels, func_models[idx]);
			if(i == 0)
				accs.push_back(acc);
			else
				accs[idx] += acc;
			idx++;
		}
		auto idx2 = 0;
		for(auto func : funcs2)
		{
			auto train_feats = get_features_from_images(train_images, func, 10);
			auto test_feats = get_features_from_images(test_images, func, 10);
			auto acc = svm_train_and_test(train_feats, train_labels, test_feats, test_labels, func_models2[idx2]);
			if(i == 0)
				accs.push_back(acc);
			else
				accs[idx] += acc;
			idx++;
			idx2++;
		}
	}
	for(auto acc : accs)
	{
		std::cout<<"0	acc:	"<<acc*100/iters<<std::endl;
		ofs<<"0		acc:	"<<acc*100/iters<<std::endl;
	}
	ofs.close();
	std::cout<<"all time: "<<t.elapsed()<<std::endl;
}