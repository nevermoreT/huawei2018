#include"LinearRegression.h"
#include<iostream>
using namespace std;
int main()
{
	vector<vector<double>> X({ {1},{3},{5},{7} });
	vector<double> y({ 2,4,6,8 });
	LinearRegression lr;
	//vector<double> coef({ 1.2,0.4 });
	//vector<double>y_p=lr.predict(X, coef);
	vector<double> coef = lr.coefficients_sgd(X, y,0.01, 0,10000);
	
	for (int i = 0; i < coef.size(); i++)
		cout << coef[i] << endl;
	system("pause");
	return 0;
}