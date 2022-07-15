#include<iostream>
using namespace std;

//设计不同大小
const int N = 1024;

int sum[N];
int b[N][N];
int a[N];

void random_init(int n, int* a)
{
	for (int i = 0; i < n; i++)
		a[i] = 1;
}

void row_major(int n, int* sum, int b[][N], int* a)
{
	int i, j;
	for (i = 0; i < n; i++)
		sum[i] = 0.0;
	for (j = 0; j < n; j++)
	{
		for (i = 0; i < n; i++)
			sum[i] += b[j][i] * a[j];
	}
}

void column_major(int n, int* sum, int b[][N], int* a)
{
	int i, j;
	for (i = 0; i < n; i++) {
		sum[i] = 0.0;
		for (j = 0; j < n; j++)
			sum[i] += b[j][i] * a[j];
	}
}

int main()
{
	random_init(N, a);
	for (int i = 0; i < N; i++)
		random_init(N, b[i]);
	for (int i = 0; i < 1; i++)
	{
		//平凡算法
		column_major(N, sum, b, a);
		//优化算法
		row_major(N, sum, b, a);
	}
	cout << sum;

}