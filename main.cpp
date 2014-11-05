#include <fstream>
#include <time.h>
#include "feature.h"
#include "classify_tools.h"
#include <tbb/task_group.h>

extern void load_image_blocks_from_path(const char * positive_path, const char * negative_path,
								 std::vector<cv::Mat> &images, std::vector<int> &labels,
								 int pos_label, int neg_label);

int main()
{
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	auto t = clock();
	load_image_blocks_from_path("D:\\img\\24\\smoke", "D:\\img\\24\\nonsmoke", images, labels, 1, -1);
	t = clock() - t;
	std::cout<<"load dataset complete! use time : "<<t/CLOCKS_PER_SEC<<"s"<<std::endl;
	std::ofstream ofs("2yuan-feature-result.txt");

	/*
	t = clock();
	//test lbp
	auto lbp_feats = get_features_from_images(images, get_lbp_gray);
	train_and_test(lbp_feats, labels, "yuan-svm-lbp.model", ofs);
	//test u-lbp
	auto u_lbp_feats = get_features_from_images(images, get_u_lbp_gray);
	train_and_test(u_lbp_feats, labels, "yuan-svm-u-lbp.model", ofs);
	//test bgc1
	auto bgc1_feats = get_features_from_images(images, get_bgc1_gray);
	train_and_test(bgc1_feats, labels, "yuan-svm-bgc1.model", ofs);
	//test bgc2
	auto bgc2_feats = get_features_from_images(images, get_bgc2_gray);
	train_and_test(bgc2_feats, labels, "yuan-svm-bgc2.model", ofs);
	//test bgc3
	auto bgc3_feats = get_features_from_images(images, get_bgc3_gray);
	train_and_test(bgc3_feats, labels, "yuan-svm-bgc3.model", ofs);
	//test gld
	auto gld_feats = get_features_from_images(images, get_gld_gray);
	train_and_test(gld_feats, labels, "yuan-svm-gld.model", ofs);
	//test rt
	auto rt_feats = get_features_from_images(images, get_rt_gray);
	train_and_test(rt_feats, labels, "yuan-svm-rt.model", ofs);
	//test rtu
	auto rtu_feats = get_features_from_images(images, get_rtu_gray);
	train_and_test(rtu_feats, labels, "yuan-svm-rtu.model", ofs);
	//test ilbp
	auto ilbp_feats = get_features_from_images(images, get_ilbp_gray);
	train_and_test(ilbp_feats, labels, "yuan-svm-ilbp.model", ofs);
	//test 3dlbp
	auto _3dlbp_feats = get_features_from_images(images, get_3dlbp_gray);
	train_and_test(_3dlbp_feats, labels, "yuan-svm-3dlbp.model", ofs);
	//test mbp
	auto mbp_feats = get_features_from_images(images, get_mbp_gray);
	train_and_test(mbp_feats, labels, "yuan-svm-mbp.model", ofs);
	//test dlbp
	auto dlbp_feats = get_features_from_images(images, get_dlbp_gray);
	train_and_test(dlbp_feats, labels, "yuan-svm-dlbp.model", ofs);
	//test idlbp
	auto idlbp_feats = get_features_from_images(images, get_idlbp_gray);
	train_and_test(idlbp_feats, labels, "yuan-svm-idlbp.model", ofs);
	//test glbp
	auto glbp_feats = get_features_from_images(images, get_glbp_gray);
	train_and_test(glbp_feats, labels, "yuan-svm-glbp.model", ofs);
	//test mts
	auto mts_feats = get_features_from_images(images, get_mts_gray);
	train_and_test(mts_feats, labels, "yuan-svm-mts.model", ofs);
	//test eoh
	auto eoh_feats = get_features_from_images(images, get_eoh_gray);
	train_and_test(eoh_feats, labels, "yuan-svm-eoh.model", ofs);
	//test sts
	auto sts_feats = get_features_from_images(images, get_sts_gray, 10);
	train_and_test(sts_feats, labels, "yuan-svm-sts.model", ofs);
	//test cslbp
	auto cslbp_feats = get_features_from_images(images, get_cslbp_gray, 10);
	train_and_test(cslbp_feats, labels, "yuan-svm-cslbp.model", ofs);
	//test cbp
	auto cbp_feats = get_features_from_images(images, get_cbp_gray, 10);
	train_and_test(cbp_feats, labels, "yuan-svm-cbp.model", ofs);
	//test csts
	auto csts_feats = get_features_from_images(images, get_csts_gray, 10);
	train_and_test(csts_feats, labels, "yuan-svm-csts.model", ofs);
	//test ltp
	auto ltp_feats = get_features_from_images(images, get_ltp_gray, 10);
	train_and_test(ltp_feats, labels, "yuan-svm-ltp.model", ofs);
	//test iltp
	auto iltp_feats = get_features_from_images(images, get_iltp_gray, 10);
	train_and_test(iltp_feats, labels, "yuan-svm-iltp.model", ofs);

	t = clock() - t;
	std::cout<<"all time: "<<t/CLOCKS_PER_SEC<<std::endl;
	ofs<<"all time: "<<t/CLOCKS_PER_SEC<<std::endl;
	*/

	t = clock();
	int length = labels.size();
	double accs[22] = {0};
	std::vector<cv::Mat> train_images, test_images;
	std::vector<int> train_labels, test_labels;
	train_images.resize(length/2);
	train_labels.resize(length/2);
	test_images.resize(length - length/2);
	test_labels.resize(length - length/2);
	for(int i = 0; i < 10; i++)
	{
		std::cout<<"iter : "<<i<<std::endl;
		srand(clock());
		int r1,r2;
		for(int i = 0; i < length / 2; i++)
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
		
		auto iti = images.begin();
		auto itl = labels.begin();
		std::copy(iti, iti+length/2, train_images.begin());
		std::copy(iti+length/2, images.end(), test_images.begin());
		std::copy(itl, itl+length/2, train_labels.begin());
		std::copy(itl+length/2, labels.end(), test_labels.begin());

		//test lbp
		auto train_lbp_feats = get_features_from_images(train_images, get_lbp_gray);
		auto test_lbp_feats = get_features_from_images(test_images, get_lbp_gray);
		accs[0] += train_and_test(train_lbp_feats, train_labels, test_lbp_feats, test_labels, "2yuan-svm-lbp.model");
		//test u-lbp
		auto train_u_lbp_feats = get_features_from_images(train_images, get_u_lbp_gray);
		auto test_u_lbp_feats = get_features_from_images(test_images, get_u_lbp_gray);
		accs[1] += train_and_test(train_u_lbp_feats, train_labels, test_u_lbp_feats, test_labels, "2yuan-svm-u-lbp.model");
		//test bgc1
		auto train_bgc1_feats = get_features_from_images(train_images, get_bgc1_gray);
		auto test_bgc1_feats = get_features_from_images(test_images, get_bgc1_gray);
		accs[2] += train_and_test(train_bgc1_feats, train_labels, test_bgc1_feats, test_labels,"2yuan-svm-bgc1.model");
		//test bgc2
		auto train_bgc2_feats = get_features_from_images(train_images, get_bgc2_gray);
		auto test_bgc2_feats = get_features_from_images(test_images, get_bgc2_gray);
		accs[3] += train_and_test(train_bgc2_feats, train_labels, test_bgc2_feats, test_labels,"2yuan-svm-bgc2.model");
		//test bgc3
		auto train_bgc3_feats = get_features_from_images(train_images, get_bgc3_gray);
		auto test_bgc3_feats = get_features_from_images(test_images, get_bgc3_gray);
		accs[4] += train_and_test(train_bgc3_feats, train_labels, test_bgc3_feats, test_labels, "2yuan-svm-bgc3.model");
		//test gld
		auto train_gld_feats = get_features_from_images(train_images, get_gld_gray);
		auto test_gld_feats = get_features_from_images(test_images, get_gld_gray);
		accs[5] += train_and_test(train_gld_feats, train_labels, test_gld_feats, test_labels, "2yuan-svm-gld.model");
		//test rt
		auto train_rt_feats = get_features_from_images(train_images, get_rt_gray);
		auto test_rt_feats = get_features_from_images(test_images, get_rt_gray);
		accs[6] += train_and_test(train_rt_feats, train_labels, test_rt_feats, test_labels,"2yuan-svm-rt.model");
		//test rtu
		auto train_rtu_feats = get_features_from_images(train_images, get_rtu_gray);
		auto test_rtu_feats = get_features_from_images(test_images, get_rtu_gray);
		accs[7] += train_and_test(train_rtu_feats, train_labels, test_rtu_feats, test_labels, "2yuan-svm-rtu.model");
		//test ilbp
		auto train_ilbp_feats = get_features_from_images(train_images, get_ilbp_gray);
		auto test_ilbp_feats = get_features_from_images(test_images, get_ilbp_gray);
		accs[8] += train_and_test(train_ilbp_feats, train_labels, test_ilbp_feats, test_labels, "2yuan-svm-ilbp.model");
		//test 3dlbp
		auto train_3dlbp_feats = get_features_from_images(train_images, get_3dlbp_gray);
		auto test_3dlbp_feats = get_features_from_images(test_images, get_3dlbp_gray);
		accs[9] += train_and_test(train_3dlbp_feats, train_labels, test_3dlbp_feats, test_labels, "2yuan-svm-3dlbp.model");
		//test mbp
		auto train_mbp_feats = get_features_from_images(train_images, get_mbp_gray);
		auto test_mbp_feats = get_features_from_images(test_images, get_mbp_gray);
		accs[10] += train_and_test(train_mbp_feats, train_labels, test_mbp_feats, test_labels, "2yuan-svm-mbp.model");
		//test dlbp
		auto train_dlbp_feats = get_features_from_images(train_images, get_dlbp_gray);
		auto test_dlbp_feats = get_features_from_images(test_images, get_dlbp_gray);
		accs[11] += train_and_test(train_dlbp_feats, train_labels, test_dlbp_feats, test_labels, "2yuan-svm-dlbp.model");
		//test idlbp
		auto train_idlbp_feats = get_features_from_images(train_images, get_idlbp_gray);
		auto test_idlbp_feats = get_features_from_images(test_images, get_idlbp_gray);
		accs[12] += train_and_test(train_idlbp_feats, train_labels, test_idlbp_feats, test_labels, "2yuan-svm-idlbp.model");
		//test glbp
		auto train_glbp_feats = get_features_from_images(train_images, get_glbp_gray);
		auto test_glbp_feats = get_features_from_images(test_images, get_glbp_gray);
		accs[13] += train_and_test(train_glbp_feats, train_labels, test_glbp_feats, test_labels, "2yuan-svm-glbp.model");
		//test mts
		auto train_mts_feats = get_features_from_images(train_images, get_mts_gray);
		auto test_mts_feats = get_features_from_images(test_images, get_mts_gray);
		accs[14] += train_and_test(train_mts_feats, train_labels, test_mts_feats, test_labels,  "2yuan-svm-mts.model");
		//test eoh
		auto train_eoh_feats = get_features_from_images(train_images, get_eoh_gray);
		auto test_eoh_feats = get_features_from_images(test_images, get_eoh_gray);
		accs[15] += train_and_test(train_eoh_feats, train_labels, test_eoh_feats, test_labels, "2yuan-svm-eoh.model");
		//test sts
		auto train_sts_feats = get_features_from_images(train_images, get_sts_gray, 10);
		auto test_sts_feats = get_features_from_images(test_images, get_sts_gray, 10);
		accs[16] += train_and_test(train_sts_feats, train_labels, test_sts_feats, test_labels, "2yuan-svm-sts.model");
		//test cslbp
		auto train_cslbp_feats = get_features_from_images(train_images, get_cslbp_gray, 10);
		auto test_cslbp_feats = get_features_from_images(test_images, get_cslbp_gray, 10);
		accs[17] += train_and_test(train_cslbp_feats, train_labels, test_cslbp_feats, test_labels, "2yuan-svm-cslbp.model");
		//test cbp
		auto train_cbp_feats = get_features_from_images(train_images, get_cbp_gray, 10);
		auto test_cbp_feats = get_features_from_images(test_images, get_cbp_gray, 10);
		accs[18] += train_and_test(train_cbp_feats, train_labels, test_cbp_feats, test_labels, "2yuan-svm-cbp.model");
		//test csts
		auto train_csts_feats = get_features_from_images(train_images, get_csts_gray, 10);
		auto test_csts_feats = get_features_from_images(test_images, get_csts_gray, 10);
		accs[19] += train_and_test(train_csts_feats, train_labels, test_csts_feats, test_labels, "2yuan-svm-csts.model");
		//test ltp
		auto train_ltp_feats = get_features_from_images(train_images, get_ltp_gray, 10);
		auto test_ltp_feats = get_features_from_images(test_images, get_ltp_gray, 10);
		accs[20] += train_and_test(train_ltp_feats, train_labels, test_ltp_feats, test_labels, "2yuan-svm-ltp.model");
		//test iltp
		auto train_iltp_feats = get_features_from_images(train_images, get_iltp_gray, 10);
		auto test_iltp_feats = get_features_from_images(test_images, get_iltp_gray, 10);
		accs[21] += train_and_test(train_iltp_feats, train_labels, test_iltp_feats, test_labels,"2yuan-svm-iltp.model");
	}
	for(int i = 0; i < 22; i++)
	{
		std::cout<<"0	acc:	"<<accs[i]/10<<std::endl;
		ofs<<"0		acc:	"<<accs[i]/10<<std::endl;
	}
	ofs.close();
	t = clock() - t;
	std::cout<<"all time: "<<t/CLOCKS_PER_SEC<<std::endl;
	std::cout<<"press any key"<<std::endl;
	getchar();

	return 0;
}