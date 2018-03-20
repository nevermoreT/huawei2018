#include "LinearRegression.h"
#include<vector>
#include<iostream>
#include<stdio.h>
using namespace std;

LinearRegression::LinearRegression()
{
}


LinearRegression::~LinearRegression()
{
}

vector<double> LinearRegression::predict(vector<vector<double>>& X, vector<double>& coef)
{
	int size = X[0].size();
	if (size != coef.size() - 1)
	{
		cout << "Wrong input:the input matrix paramer is " << size << "but the coef is " << coef.size() - 1 << endl;
		exit(0);
	}
	vector<double> y(X.size());
	//add the bias
	
	for (int i = 0; i < X.size(); i++)
	{
		y[i] = 0;
		for (int j = 0; j < coef.size()-1; j++)
		{
			y[i] += X[i][j] * coef[j];
		}
		y[i] += coef[coef.size() - 1]; //add the bias
		
	}
	return y;
}

vector<double> LinearRegression::coefficients_sgd(vector<vector<double>>& X, vector<double>& y, double l_rate, double lambda,int n_epoch)
{
	vector<double> coef(X[0].size() + 1);
	for (vector<double>::iterator it = coef.begin(); it != coef.end(); it++)
	{
		*it = 0; //初始化参数为零
	}
	for (int i = 1; i <= n_epoch; i++)
	{
		vector<double> predict_y = this->predict(X, coef);
		
		int sgd_x = rand() % X.size();//随机使用一个向量更新梯度
		
		double error = predict_y[sgd_x] - y[sgd_x];
		for (int j=0;j<coef.size();j++)
		{
			if (j == coef.size() - 1)
			{
				coef[j] = coef[j] * (1 - l_rate * lambda) - l_rate * error * 1; //处理bias
			}
			else
			{
				coef[j] = coef[j] * (1 - l_rate * lambda) - l_rate * error*X[sgd_x][j];  //sgd更新梯度
			}
		}
		/*
		printf(">epoch=%d, lrate=%.3f, error=%.3f\n", i, l_rate,error);
		printf("coef:");
		for (int i = 0; i < coef.size(); i++)
			printf("%f ", coef[i]);
		printf("\n");*/
	}
	this->coef = coef;
	return coef;
}

