#include "bwlabel.h"

int number_of_runs(const cv::Mat in)
{
	const int rows = in.rows;
	const int cols = in.cols;
	int result = 0;
	for(int row = 0; row < rows; row++)
	{
		const uchar * p_row = in.ptr<uchar>(row);
		if(p_row[0] != 0)
			result++;
		for(int col = 1; col < cols; col++)
		{
			if(p_row[col] != 0 && p_row[col-1] == 0)
				result++;
		}
	}
	return result;
}
void fill_run_vectors(const cv::Mat in, int sc[], int ec[], int r[])
{
	const int rows = in.rows;
	const int cols = in.cols;
	int idx = 0;
	for(int row = 0; row < rows; row++)
	{
		const uchar * p_row = in.ptr<uchar>(row);
		int prev = 0;
		for(int col = 0; col < cols; col++)
		{
			if(p_row[col] != prev)
			{
				if(prev == 0)
				{
					sc[idx] = col;
					r[idx] = row;
					prev = 1;
				}
				else
				{
					ec[idx++] = col - 1;
					prev = 0;
				}
			}
			if(col == cols-1 && prev == 1)
			{
				ec[idx++] = col;
			}
		}
	}
}
void first_pass(const int sc[], const int ec[], const int r[],int labels[], const int num_runs, const int mode)
{
	int cur_row = 0;
	int next_label = 1;
	int first_run_on_prev_row = -1;
	int last_run_on_prev_row = -1;
	int first_run_on_this_row = 0;
	int offset = 0;
	int * equal_i = new int[num_runs];
	int * equal_j = new int[num_runs];
	int equal_idx = 0;
	if(mode == 8)
		offset = 1;
	for(int k = 0; k < num_runs; k++)
	{
		if(r[k] == cur_row + 1)
		{
			cur_row += 1;
			first_run_on_prev_row = first_run_on_this_row;
			first_run_on_this_row = k;
			last_run_on_prev_row = k - 1;
		}
		else if(r[k] > cur_row + 1)
		{
			first_run_on_prev_row = -1;
			last_run_on_prev_row = -1;
			first_run_on_this_row = k;
			cur_row = r[k];
		}
		if(first_run_on_prev_row >= 0)
		{
			int p = first_run_on_prev_row;
			while(p <= last_run_on_prev_row && sc[p] <= (ec[k] + offset))
			{
				if(sc[k] <= ec[p] + offset)
				{
					if(labels[k] == 0)
						labels[k] = labels[p];
					else if(labels[k] != labels[p])
					{
						//labels[p] = labels[k];
						equal_i[equal_idx] = labels[k];
						equal_j[equal_idx] = labels[p];
						equal_idx += 1;
					}
				}
				p += 1;
			}
		}
		if(labels[k] == 0)
		{
			labels[k] = next_label++;
		}
	}
	/////////////////////// process labels
	for(int i = 0; i < equal_idx; i++)
	{
		int max_label = equal_i[i] > equal_j[i] ? equal_i[i] : equal_j[i];
		int min_label = equal_i[i] < equal_j[i] ? equal_i[i] : equal_j[i];
		for(int j = 0; j < num_runs; j++)
		{
			if(labels[j] == max_label)
				labels[j] = min_label;
		}
	}
	delete [] equal_i;
	delete [] equal_j;
	/////////////////////process ignore labels
	int * hist = new int[next_label];
	int * non_labels = new int[next_label];
	memset(hist, 0, sizeof(int)*next_label);
	int non_num = 0;
	for(int i = 0; i < num_runs; i++)
	{
		hist[labels[i]]++;
	}
	for(int i = 1; i < next_label; i++)
	{
		if(hist[i] == 0)
			non_labels[non_num++] = i;
	}
	for(int j = 0; j < num_runs; j++)
	{
		int k = labels[j];
		for(int i = non_num-1; i >= 0; i--)
		{
			if(k > non_labels[i])
			{
				labels[j] -= (i+1);
				break;
			}
		}
	}
	delete [] hist;
	delete [] non_labels;
}

cv::Mat bwlabel(const cv::Mat in, int * num, const int mode)
{
	const int num_runs = number_of_runs(in);
	int * sc = new int[num_runs];
	int * ec = new int[num_runs];
	int * r = new int[num_runs];
	int * labels = new int[num_runs];
	memset(labels, 0, sizeof(int)*num_runs);
	fill_run_vectors(in, sc, ec, r);
	first_pass(sc, ec, r, labels, num_runs, mode);
	cv::Mat result = cv::Mat::zeros(in.size(), CV_8UC1);

	int number = 0;
	for(int i = 0; i < num_runs; i++)
	{
		uchar * p_row = result.ptr<uchar>(r[i]);
		for(int j = sc[i]; j <= ec[i]; j++)
			p_row[j] = labels[i];
		if(number < labels[i])
			number = labels[i];
	}
	if(num != NULL)
		*num = number;
	delete [] sc;
	delete [] ec;
	delete [] r;
	delete [] labels;
	return result;
}
