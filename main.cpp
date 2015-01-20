#include <fstream>
#include <time.h>
#include "feature.h"
#include "classify_tools.h"
#include "dataset_tools.h"
#include <tbb/task_group.h>
#include <tiny_cnn/tiny_cnn.h>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void svm_train_test();
void mlp_train_test();
void d_svm_train_test();
void d_mlp_train_test();

extern void cnn_test();

///////////////////////////////// global
p_static_feature_function funcs[] = {get_lbp_gray, get_u_lbp_gray, get_bgc1_gray, get_bgc2_gray, get_bgc3_gray,
		get_rt_gray, get_rtu_gray, get_ilbp_gray, get_3dlbp_gray,get_mbp_gray, get_dlbp_gray, get_idlbp_gray,
		get_glbp_gray, get_mts_gray, get_eoh_gray};
const char * func_names[] = {"get_lbp_gray","get_u_lbp_gray", "get_bgc1_gray", "get_bgc2_gray", "get_bgc3_gray",
		"get_rt_gray", "get_rtu_gray", "get_ilbp_gray", "get_3dlbp_gray", "get_mbp_gray", "get_dlbp_gray", "get_idlbp_gray",
		"get_glbp_gray", "get_mts_gray", "get_eoh_gray"};
const char * func_models[] = {"lbp_gray.model","u_lbp_gray.model", "bgc1_gray.model", "bgc2_gray.model", "bgc3_gray.model",
		"rt_gray.model", "rtu_gray.model", "ilbp_gray.model", "3dlbp_gray.model", "mbp_gray.model", "dlbp_gray.model", "idlbp_gray.model",
		"glbp_gray.model", "mts_gray.model", "eoh_gray.model"};
const char * func_weights[] = {"lbp_gray.weights","u_lbp_gray.weights", "bgc1_gray.weights", "bgc2_gray.weights", "bgc3_gray.weights",
		"rt_gray.weights", "rtu_gray.weights", "ilbp_gray.weights", "3dlbp_gray.weights", "mbp_gray.weights", "dlbp_gray.weights", "idlbp_gray.weights",
		"glbp_gray.weights", "mts_gray.weights", "eoh_gray.weights"};
p_static_feature_function_with_delta funcs2[] = {get_sts_gray,get_cslbp_gray,get_cbp_gray,get_csts_gray,get_ltp_gray,get_iltp_gray};
const char * func_names2[] = {"get_sts_gray","get_cslbp_gray","get_cbp_gray","get_csts_gray","get_ltp_gray","get_iltp_gray"};
const char * func_models2[] = {"sts_gray.model","cslbp_gray.model","cbp_gray.model","csts_gray.model","ltp_gray.model","iltp_gray.model"};
const char * func_weights2[] = {"sts_gray.weights","cslbp_gray.weights","cbp_gray.weights","csts_gray.weights","ltp_gray.weights","iltp_gray.weights"};

p_dynamic_feature_function dfuncs[] = {get_eoh_top, get_lbp_top, get_bgc3_top, get_rtu_top};
const char * dfunc_names[] = {"eoh_top", "lbp_top", "bgc3_top", "rtu_top"};
const char * dfunc_models[] = {"eoh_top.model", "lbp_top.model", "bgc3_top.model", "rtu_top.model"};
const char * dfunc_weights[] = {"eoh_top.weights", "lbp_top.weights", "bgc3_top.weights", "rtu_top.weights"};

int main()
{
	//svm_train_test();
	//mlp_train_test();
	//d_svm_train_test();
	//cnn_test();
	d_mlp_train_test();
	std::cout<<"complete press any key out"<<std::endl;
	getchar();
}
void mlp_train_test()
{
	//load dataset
	std::vector<cv::Mat> pos_images, neg_images;
	
	std::vector<cv::Mat> train_images;
	std::vector<cv::Mat> test_images;
	std::vector<label_t> train_labels;
	std::vector<label_t> test_labels;
	
	boost::timer t;
	load_image_blocks_from_path("F:\\DataSet\\smoke2\\24\\smoke", &pos_images);
	load_image_blocks_from_path("F:\\DataSet\\smoke2\\24\\nonsmoke", &neg_images);
	//load_image_blocks_from_path("D:\\wzj\\dataset\\smoke\\24\\smoke", pos_images); //·þÎñÆ÷Â·¾¶
	//load_image_blocks_from_path("D:\\wzj\\dataset\\smoke\\24\\nonsmoke", neg_images);
	rand_split_train_test(pos_images, neg_images, &train_images, &train_labels, &test_images, &test_labels, 0.2);
	//
	int test_pos = 0;
	int test_neg = 0;
	for(auto & y : test_labels)
	{
		if(y == 1)
			test_pos++;
		else
			test_neg++;
	}
	std::cout << "test pos : " << test_pos << " ,test neg : " << test_neg << std::endl;
	std::cout<<"load dataset complete! use time : "<<t.elapsed()<<"s"<<std::endl;
	std::ofstream ofs("24-mlp-result.txt");
	
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
		nn.save_weights(func_weights[idx]);
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

		nn.train(train_feats, train_labels, 1, 20, on_enumerate_data, on_enumerate_epoch);
		std::cout<<func_names2[idx]<<std::endl;
		nn.test(test_feats, test_labels).print_detail(std::cout);
		ofs<<func_names2[idx]<<std::endl;
		nn.test(test_feats, test_labels).print_detail(ofs);
		nn.save_weights(func_weights2[idx]);
		idx++;
	}
	ofs.close();
}
void svm_train_test()
{
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	std::vector<cv::Mat> train_images, test_images;
	std::vector<int> train_labels, test_labels;
	boost::timer t;
	load_image_blocks_from_path("d:\\wzj\\dataset\\smoke\\24\\smoke", &images, &labels, 1);
	load_image_blocks_from_path("d:\\wzj\\dataset\\smoke\\24\\nonsmoke", &images, &labels, 0);
	rand_order(images, labels);
	train_test_split(images, labels, &train_images, &train_labels, &test_images, &test_labels, 0.2);
	std::cout<<"load dataset complete! use time : "<<t.elapsed()<<"s"<<std::endl;
	std::ofstream ofs("24-svm-result.txt");
	//
	
	std::vector<double> accs;
	//iter 10s
	t.restart();
		auto idx = 0;
		for(auto func : funcs)
		{
			auto train_feats = get_features_from_images(train_images, func);
			auto test_feats = get_features_from_images(test_images, func);
			auto acc = svm_train_and_test(train_feats, train_labels, test_feats, test_labels, func_models[idx]);
			std::cout << func_names[idx] << " : " << acc << std:: endl;
			ofs << func_names[idx] << " : " << acc << std:: endl;
			idx++;
		}
		auto idx2 = 0;
		for(auto func : funcs2)
		{
			auto train_feats = get_features_from_images(train_images, func, 10);
			auto test_feats = get_features_from_images(test_images, func, 10);
			auto acc = svm_train_and_test(train_feats, train_labels, test_feats, test_labels, func_models2[idx2]);
			std::cout << func_names2[idx2] << " : " << acc << std:: endl;
			ofs << func_names2[idx2] << " : " << acc << std:: endl;
			idx2++;
		}
	ofs.close();
	std::cout<<"all time: "<<t.elapsed()<<std::endl;
}

void d_svm_train_test()
{
	std::vector<mat3_t> images;
	std::vector<mat3_t> pos_x, neg_x, train_x, test_x;
	std::vector<label_t>train_y, test_y;

	int pos_length, neg_length;

	//load_d_from_video("F:\\DataSet\\smoke2\\v24\\smoke", images);
	load_d_from_video("D:\\wzj\\dataset\\smoke\\v24\\smoke", &images); //
	for(int i = 0; i < images.size(); i++)
	{
		split_d_smokes2(images[i], &pos_x, 10, 5);
	}
	images.clear(); //clear 
	//load_d_from_video("F:\\DataSet\\smoke2\\v24\\nonsmoke", images);
	load_d_from_video("D:\\wzj\\dataset\\smoke\\v24\\nonsmoke", &images); //
	for(int i = 0; i < images.size(); i++)
	{
		split_d_smokes2(images[i], &neg_x, 10, 5);
	}
	images.clear();
	rand_split_train_test(pos_x, neg_x, &train_x, &train_y, &test_x, &test_y, 0.2);
	pos_x.clear();
	neg_x.clear();

	std::ofstream ofs("24-top-result.txt");

	int idx = 0;
	for(auto & func : dfuncs)
	{
		auto train_feats = get_dynamic_features(train_x, func);
		auto test_feats  = get_dynamic_features(test_x, func);
		auto acc = svm_train_and_test(train_feats, train_y, test_feats, test_y, dfunc_models[idx]);
		std::cout << dfunc_names[idx] << " : " << acc << std::endl;
		ofs << dfunc_names[idx] << " : " << acc << std::endl;
		idx++;
	}
	ofs.close();
}

void d_mlp_train_test()
{
	std::vector<mat3_t> images;
	std::vector<mat3_t> pos_x, neg_x, train_x, test_x;
	std::vector<label_t>train_y, test_y;

	int pos_length, neg_length;

	load_d_from_video("F:\\DataSet\\smoke2\\v24\\smoke", &images);
	//load_d_from_video("D:\\wzj\\dataset\\smoke\\v24\\smoke", images); //
	for(int i = 0; i < images.size(); i++)
	{
		split_d_smokes2(images[i], &pos_x, 10, 5);
	}
	images.clear(); //clear 
	load_d_from_video("F:\\DataSet\\smoke2\\v24\\nonsmoke", &images);
	//load_d_from_video("D:\\wzj\\dataset\\smoke\\v24\\nonsmoke", images); //
	for(int i = 0; i < images.size(); i++)
	{
		split_d_smokes2(images[i], &neg_x, 10, 5);
	}
	images.clear();
	rand_split_train_test(pos_x, neg_x, &train_x, &train_y, &test_x, &test_y, 0.2);
	pos_x.clear();
	neg_x.clear();

	std::ofstream ofs("24-top-result.txt");

	int idx = 0;
	for(auto & func : dfuncs)
	{
		auto train_feats = get_dynamic_features(train_x, func);
		auto test_feats  = get_dynamic_features(test_x, func);
		//construct mlp
		const int num_input = train_feats[0].size();
		const int num_hidden_units = 30;
		int num_units[] = { num_input, num_hidden_units, 2 };
		auto nn = make_mlp<mse, gradient_descent_levenberg_marquardt, tan_h>(num_units, num_units + 3);

		//train mlp
		nn.optimizer().alpha = 0.005;
		boost::progress_display disp(train_feats.size());
		boost::timer t;
		// create callback
		auto on_enumerate_epoch = [&](){
			std::cout << t.elapsed() << "s elapsed." << std::endl;
			tiny_cnn::result res = nn.test(test_feats, test_y);
			std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;
			nn.optimizer().alpha *= 0.85; // decay learning rate
			nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);
			disp.restart(train_feats.size());
			t.restart();
		};
		auto on_enumerate_data = [&](){ 
			++disp; 
		};  

		nn.train(train_feats, train_y, 1, 20, on_enumerate_data, on_enumerate_epoch);
		std::cout<<func_names[idx]<<std::endl;
		nn.test(test_feats, test_y).print_detail(std::cout);
		ofs<<func_names[idx]<<std::endl;
		nn.test(test_feats, test_y).print_detail(ofs);
		nn.save_weights(dfunc_weights[idx]);
		idx++;
	}
	ofs.close();
}