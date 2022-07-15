#include<iostream>
using namespace std;

//使用SSE intrinsics所需的头文件
#include <xmmintrin.h>  //SSE
#include <emmintrin.h>  //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h>  //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h>  //SSSE4.2
#include <immintrin.h> //AVX、AVX2、AVX-512

const int N = 10;//矩阵列数
const int r = 6;//消元子数
const int e = 3;//被消元行数

int R[N][N] = { 0 };//所有消元子构成的集合R
int E[N][N] = { 0 };//所有被消元行构成的数组E
int lp[N];//首项
int lpE[N];//lp(E[i]):被消元行第i行的首项
int t = r;

//E[i]!=0
bool Nonzero(int* E, int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += E[i];
	if (sum == 0)
		return false;
	else
		return true;
}

//R[lp(E[i])!=NULL
bool RlpEi_NULL(int x)
{
	for (int i = 0; lp[i] != -1; i++)
	{
		if (lp[i] == x)
			return true;
	}
	return false;
}

//Groebner基的普通高斯消元
//所有消元子构成的集合R
//所有被消元行构成的数组E
//被消元行数m
void GroebnerCpp(int R[N][N], int E[N][N], int e)
{
	//由于特殊高斯消元顶层为最大行数，由小至大排
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (!Nonzero(E[i], N))
		{
			if (RlpEi_NULL(lpE[N - 1 - i]))
			{
				for (int k = N - 1; k > -1; k--)
				{
					E[i][k] = E[i][k] ^ R[lpE[N - 1 - i]][k];
				}
				for (int i = N - 1; i > -1; i--)
				{
					if (E[i] != 0)
					{
						lpE[N - 1 - i] = i;
						return;
					}
				}
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[t] = lpE[N - i - 1];
					t++;
					for (int j = 0; j < N; j++)
						R[lp[t - 1]][j] = E[i][j];
					goto L1;
				}
			}
		}
	L1:i--;
	}
}

//SSE
void Groebner_SIMD_SSE(int R[N][N], int E[N][N], int e)
{
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (Nonzero(E[i], N))
		{
			if (RlpEi_NULL(lpE[N - 1 - i]))
			{
				int k;
				for (k = 0; k + 3 < N; k += 4)
				{
					__m128i eik = _mm_load_si128((__m128i*)(E[i] + k));
					__m128i rik = _mm_load_si128((__m128i*)(R[lpE[N - 1 - i]] + k));
					eik = _mm_xor_si128(eik, rik);
					_mm_store_si128((__m128i*)(E[i] + k), eik);
					_mm_store_si128((__m128i*)(R[lpE[N - 1 - i]] + k), rik);
				}
				for (; k < N; k++)
				{
					E[i][k] = E[i][k] ^ R[lpE[N - 1 - i]][k];
				}
				for (int i = N - 1; i > -1; i--)
				{
					if (E[i] != 0)
					{
						lpE[N - 1 - i] = i;
						return;
					}
				}
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[t] = lpE[N - i - 1];
					t++;
					int j;
					for (j = 0; j + 3 < N; j += 4)
					{
						__m128i rtj = _mm_load_si128((__m128i*)(E[i] + j));
						_mm_store_si128((__m128i*)(R[lp[t - 1]] + j), rtj);
					}
					for (; j < N; j++)
						R[lp[t - 1]][j] = E[i][j];
					for (int j = 0; j < N; j++)
						R[lp[t - 1]][j] = E[i][j];
					goto L2;
				}
			}
		}
	L2:i--;
	}
}

//AVX
void Groebner_SIMD_AVX(int R[N][N], int E[N][N], int e)
{
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (Nonzero(E[i], N))
		{
			if (RlpEi_NULL(lpE[N - 1 - i]))
			{
				int k;
				for (k = 0; k + 7 < N; k += 8)
				{
					__m256i eik = _mm256_load_si256((__m256i*)(E[i] + k));
					__m256i rik = _mm256_load_si256((__m256i*)(R[lpE[N - 1 - i]] + k));
					eik = _mm256_xor_si256(eik, rik);
					_mm256_store_si256((__m256i*)(E[i] + k), eik);
					_mm256_store_si256((__m256i*)(R[lpE[N - 1 - i]] + k), rik);
				}
				for (; k < N; k++)
				{
					E[i][k] = E[i][k] ^ R[lpE[N - 1 - i]][k];
				}
				for (int i = N - 1; i > -1; i--)
				{
					if (E[i] != 0)
					{
						lpE[N - 1 - i] = i;
						return;
					}
				}
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[t] = lpE[N - i - 1];
					t++;
					int j;
					for (j = 0; j + 7 < N; j += 8)
					{
						__m256i rtj = _mm256_load_si256((__m256i*)(E[i] + j));
						_mm256_store_si256((__m256i*)(R[lp[t - 1]] + j), rtj);
					}
					for (; j < N; j++)
						R[lp[t - 1]][j] = E[i][j];
					for (int j = 0; j < N; j++)
						R[lp[t - 1]][j] = E[i][j];
					goto L3;
				}
			}
		}
	L3:i--;
	}
}

int main()
{

	return 0;
}