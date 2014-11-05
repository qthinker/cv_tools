#include "classify_tools.h"
#include <fstream>
#include <io.h>

std::vector<feature_t> get_features_from_images(std::vector<cv::Mat> &images, const p_func_feature_extract func)
{
	std::vector<feature_t> result;
	for(auto img : images)
	{
		result.push_back(func(img));
	}
	return result;
}

std::vector<feature_t> get_features_from_images(std::vector<cv::Mat> &images, const p_func_feature_extract_with_delta func, int delta)
{
	std::vector<feature_t> result;
	for(auto img : images)
	{
		result.push_back(func(img, delta));
	}
	return result;
}

svm_problem * fill_problem(std::vector<feature_t> feats, std::vector<int> labels)
{
	assert(feats.size() == labels.size());
	int length = labels.size();
	int dim = feats[0].size();
	svm_problem * prob = (svm_problem *)malloc(sizeof(svm_problem));
	prob->l = length;
	prob->y = (double *)malloc(sizeof(double) * length);
	prob->x = (svm_node **)malloc(sizeof(svm_node*) * length);
	
	for(int i = 0; i < length; i++)
	{
		prob->y[i] = labels[i];
		int len = 0;
		for(auto k : feats[i])
		{
			if(k != 0)
				len++;
		}
		svm_node * x = (svm_node *)malloc(sizeof(svm_node) * (len+1));
		int idx = 0;
		for(auto k : feats[i])
			if(k != 0)
			{
				x[idx].value = k;
				x[idx].index = idx + 1; 
				idx++;
			}
		x[len].index = -1;
		prob->x[i] = x;
	}
	return prob;
}
void free_problem(svm_problem *prob)
{
	free(prob->y);
	for(int i = 0; i < prob->l; i++)
	{
		free(prob->x[i]);
	}
	free(prob->x);
	free(prob);
}
svm_parameter * fill_parameter(double gamma, int C)
{
	svm_parameter * param = (svm_parameter *)malloc(sizeof(svm_parameter));
	param->svm_type = C_SVC;
	param->kernel_type = RBF;
	param->degree = 3;
	param->gamma = gamma;
	param->coef0 = 0;
	param->nu = 0.5;
	param->cache_size = 100;
	param->C = C;
	param->eps = 1e-3;
	param->p = 0.1;
	param->shrinking = 1;
	param->probability = 0;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;
	return param;
}

double test_acc(svm_problem * prob, svm_model * model)
{
	double success = 0;
	for(int i = 0; i < prob->l; i++)
	{
		double y = svm_predict(model, prob->x[i]);
		if(y == prob->y[i])
			success++;
	}
	success /= prob->l;
	return success;
}

void train_and_test(std::vector<feature_t> feats, std::vector<int> labels, const char * model_file, std::ofstream &ofs)
{
	svm_model * model;
	if(_access(model_file, 0) == -1)
	{
		auto gamma = 1.0 / feats[0].size();
		auto param = fill_parameter(gamma);
		auto prob = fill_problem(feats, labels);
		model = svm_train(prob, param);
		svm_save_model(model_file, model);
		auto acc = test_acc(prob, model);
		svm_destroy_param(param);
		free_problem(prob);
		std::cout<<model_file<<"  acc: "<<acc*100<<std::endl;
		ofs<<model_file<<"  acc: "<<acc*100<<std::endl;
	}
	else
	{
		model = svm_load_model(model_file);
		auto acc = test_acc(feats, labels, model);
		std::cout<<model_file<<"  acc: "<<acc*100<<std::endl;
		ofs<<model_file<<"  acc: "<<acc*100<<std::endl;
	}
	//free
	svm_free_and_destroy_model(&model);
}

void train_and_test(std::vector<feature_t> feats, std::vector<int> labels, const char * model_file)
{
	svm_model * model;
	if(_access(model_file, 0) == -1)
	{
		//auto gamma = 1.0 / feats[0].size();
		auto param = fill_parameter();
		auto prob = fill_problem(feats, labels);
		model = svm_train(prob, param);
		svm_save_model(model_file, model);
		auto acc = test_acc(prob, model);
		svm_destroy_param(param);
		free_problem(prob);
		std::cout<<model_file<<"  acc: "<<acc*100<<std::endl;
	}
	else
	{
		model = svm_load_model(model_file);
		auto acc = test_acc(feats, labels, model);
		std::cout<<model_file<<"  acc: "<<acc*100<<std::endl;
	}
	//free
	svm_free_and_destroy_model(&model);
}
//
double train_and_test(std::vector<feature_t> train_feats, std::vector<int> train_labels,std::vector<feature_t> test_feats, std::vector<int> test_labels, const char * model_file)
{
	svm_model * model;
	auto gamma = 1.0 / train_feats[0].size();
	auto param = fill_parameter(gamma, 100);
	auto prob = fill_problem(train_feats, train_labels);
	model = svm_train(prob, param);
	//svm_save_model(model_file, model);
	auto acc = test_acc(test_feats, test_labels, model);
	svm_destroy_param(param);
	free_problem(prob);
	//free
	svm_free_and_destroy_model(&model);
	std::cout<<model_file<<"  acc: "<<acc*100<<std::endl;
	return acc;
}

double test_acc(std::vector<feature_t> feats, std::vector<int> labels, svm_model * model)
{
	double success = 0;
	assert(feats.size() == labels.size());
	int length = labels.size();
	for(int i = 0; i < length; i++)
	{
		int len = 0;
		for(auto k : feats[i])
		{
			if(k != 0)
				len++;
		}
		svm_node * x = (svm_node *)malloc(sizeof(svm_node) * (len+1));
		int idx = 0;
		for(auto k : feats[i])
			if(k != 0)
			{
				x[idx].value = k;
				x[idx].index = idx + 1; 
				idx++;
			}
		x[len].index = -1;
		auto y = svm_predict(model, x);
		if(y == (double)labels[i])
			success++;
		free(x);
	}
	return success / length;
}