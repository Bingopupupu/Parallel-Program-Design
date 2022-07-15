#include<iostream>
using namespace std;

const int N = 2048;

int sum;
int a[N];

void random_init(int n, int* a)
{
	for (int i = 0; i < n; i++)
		//a[i] = rand() / int(RAND_MAX);
		a[i] = 1;
}

//链
void chain(int n)
{
	int i;
	for (i = 0; i < n; i++)
		sum += a[i];
}

// 多链路式
void Multipath(int n)
{
	int sum1 = 0, sum2 = 0, i;
	for (i = 0; i < n; i += 2)
	{
		sum1 += a[i];
		sum2 += a[i + 1];
	}
	sum = sum1 + sum2;
}

//递归：
//1. 将给定元素两两相加，得到n / 2个中间结果;
//2. 将上一步得到的中间结果两两相加，得到n / 4个中间结果;
//3. 依此类推，log(n)个步骤后得到一个值即为最终结果。
//递归函数，优点是简单，缺点是递归函数调用开销较大
void recursion(int n)
{
	int i;
	if (n == 1)
		return;
	else
	{
		for (i = 0; i < n / 2; i++)
			a[i] += a[n - i - 1];
		n = n / 2;
		// a[0]为最终结果
		sum = a[0];
		recursion(n);
	}
}

// 二重循环
void DoubleCycle(int n)
{
	int m, i;
	for (m = n; m > 1; m /= 2) // log(n)个步骤
		for (i = 0; i < m / 2; i++)
			a[i] = a[i * 2] + a[i * 2 + 1]; // 相邻元素相加连续存储到数组最前面
		 // a[0]为最终结果
	sum = a[0];
}

int main()
{
	random_init(N, a);
	for (int i = 0; i < 500; i++)
	{
		chain(N);
		Multipath(N);
		//recursion(N);
		//DoubleCycle(N);
	}
	cout << sum;
}
