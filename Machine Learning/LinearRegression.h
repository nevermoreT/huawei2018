#include<vector>
#ifndef _LinearRegression_H_
#define _LinearRegression_H_
using namespace std;
class LinearRegression
{
public:
	LinearRegression();
	~LinearRegression();
	vector<double> predict(vector <vector<double>> &X,vector<double> &coef);
	vector<double> coefficients_sgd(vector <vector<double>> &X, vector<double> &y, double l_rate, double lambda,int n_epoch);
private:
	vector<double> coef;
};
#endif
