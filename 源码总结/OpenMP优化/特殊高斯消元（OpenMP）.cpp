#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<tmmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<immintrin.h>
//#include<arm_neon.h>

#include<Windows.h>
#include<omp.h>

using namespace std;

const int N = 2632;//矩阵列数
const int r = 1226;//消元子数
const int e = 453;//被消元行数
int R[N][N] = { 0 };//消元子
int E[N][N] = { 0 };//被消元行
int txt_r[N][N];
int txt_e[N][N];
int lp[N];//消元子首项
int lpE[N];//被消元行首项
int temp = r;
//线程数
const int thread_count = 4;
const int chunk_size = 24;
//被消元行的首项随否在消元子的首项中
bool is_in_lp(int index)
{
	int i = 0;
	while (lp[i] != -1)
	{
		if (lp[i] == index)
			return true;
		i++;
	}
	return false;
}

//判断被消元行某一行是不是全是0
bool is_not_zero(int* E, int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += E[i];
	if (sum)
		return true;
	else
		return false;
}
//更新被消元行的首项位置
void Reset(int* E, int* lpE, int n)
{
	for (int i = N - 1; i > -1; i--)
	{
		if (E[i] != 0)
		{
			lpE[n] = i;
			return;
		}
	}
}

//Groebner基的普通高斯消元
void Groebner_ordinary(int R[N][N], int E[N][N], int e)
{
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (is_not_zero(E[i], N))
		{
			if (is_in_lp(lpE[N - 1 - i]))
			{
				for (int k = N - 1; k > -1; k--)
				{
					E[i][k] = E[i][k] ^ R[lpE[N - 1 - i]][k];
				}
				Reset(E[i], lpE, N - 1 - i);
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[temp] = lpE[N - i - 1];
					temp++;
					for (int j = 0; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					goto L1;
				}
			}
		}
	L1:i--;
	}
}

//基于SSE指令的Groebner基的高斯消元
void Groebner_SIMD_SSE(int R[N][N], int E[N][N], int e)
{
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (is_not_zero(E[i], N))
		{
			if (is_in_lp(lpE[N - 1 - i]))
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
				Reset(E[i], lpE, N - 1 - i);
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[temp] = lpE[N - i - 1];
					temp++;
					int j;
					for (j = 0; j + 3 < N; j += 4)
					{
						__m128i rtj = _mm_load_si128((__m128i*)(E[i] + j));
						_mm_store_si128((__m128i*)(R[lp[temp - 1]] + j), rtj);
					}
					for (; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					for (int j = 0; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					goto L2;
				}
			}
		}
	L2:i--;
	}
}

//基于AVX指令的Groebner基的高斯消元
void Groebner_SIMD_AVX(int R[N][N], int E[N][N], int e)
{

	for (int i = N - 1; i > N - 1 - e;)
	{
		while (is_not_zero(E[i], N))
		{
			if (is_in_lp(lpE[N - 1 - i]))
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
				Reset(E[i], lpE, N - 1 - i);
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[temp] = lpE[N - i - 1];
					temp++;
					int j;
					for (j = 0; j + 7 < N; j += 8)
					{
						__m256i rtj = _mm256_load_si256((__m256i*)(E[i] + j));
						_mm256_store_si256((__m256i*)(R[lp[temp - 1]] + j), rtj);
					}
					for (; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					for (int j = 0; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					goto L3;
				}
			}
		}
	L3:i--;
	}
}

//基于Neon指令的Grobner基的高斯消元
//void Groebner_SIMD_Neon(int R[N][N], int E[N][N], int e)
//{
//	for (int i = N - 1; i > N - 1 - e;)
//	{
//		while (is_not_zero(E[i], N))
//		{
//			if (is_in_lp(lpE[N - 1 - i]))
//			{
//				int k;
//				for (k = 0; k + 3 < N; k += 4)
//				{
//					int32x4_t eik = vld1q_s32(E[i] + k);
//					int32x4_t rik = vld1q_s32(R[lpE[N - 1 - i]] + k);
//					eik = vornq_s32(eik, rik);
//					vst1q_s32(E[i] + k, eik);
//					vst1q_s32(R[lpE[N - 1 - i]] + k, rik);
//				}
//				for (; k < N; k++)
//				{
//					E[i][k] = E[i][k] ^ R[lpE[N - 1 - i]][k];
//				}
//				Reset(E[i], lpE, N - 1 - i);
//			}
//			else
//			{
//				if (lpE[N - 1 - i] != -1)
//				{
//					lp[temp] = lpE[N - i - 1];
//					temp++;
//					int j;
//					for (j = 0; j + 3 < N; j += 4)
//					{
//						int32x4_t rtj = vld1q_s32(E[i]+j);
//						vst1q_s32(R[lp[temp - 1]] + j, rtj);
//					}
//					for (; j < N; j++)
//						R[lp[temp - 1]][j] = E[i][j];
//					for (int j = 0; j < N; j++)
//						R[lp[temp - 1]][j] = E[i][j];
//					goto L3;
//				}
//			}
//		}
//	L3:i--;
//	}
//}


//omp版本Groebner基高斯消元
void Groebner_omp(int R[N][N], int E[N][N], int e)
{
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (is_not_zero(E[i], N))
		{
			if (is_in_lp(lpE[N - 1 - i]))
			{
#pragma omp for schedule(dynamic,chunk_size)
				for (int k = N - 1; k > -1; k--)
				{
					E[i][k] = E[i][k] ^ R[lpE[N - 1 - i]][k];
				}
				Reset(E[i], lpE, N - 1 - i);
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[temp] = lpE[N - i - 1];
					temp++;
#pragma omp for schedule(dynamic,chunk_size)
					for (int j = 0; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					goto L1;
				}
			}
		}
	L1:i--;
	}
}

//omp版本基于AVX指令的Groebner基的高斯消元
void Groebner_omp_AVX(int R[N][N], int E[N][N], int e)
{

	for (int i = N - 1; i > N - 1 - e;)
	{
		while (is_not_zero(E[i], N))
		{
			if (is_in_lp(lpE[N - 1 - i]))
			{
				int k;
#pragma omp for schedule(dynamic,chunk_size)
				for (k = 0; k < N - 7; k += 8)
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
				Reset(E[i], lpE, N - 1 - i);
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[temp] = lpE[N - i - 1];
					temp++;
					int j;
#pragma omp for schedule(dynamic,chunk_size)
					for (j = 0; j < N - 7; j += 8)
					{
						__m256i rtj = _mm256_load_si256((__m256i*)(E[i] + j));
						_mm256_store_si256((__m256i*)(R[lp[temp - 1]] + j), rtj);
					}
					for (; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					for (int j = 0; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					goto L3;
				}
			}
		}
	L3:i--;
	}
}

//omp版本基于SSE指令的Groebner基的高斯消元
void Groebner_omp_SSE(int R[N][N], int E[N][N], int e)
{
	for (int i = N - 1; i > N - 1 - e;)
	{
		while (is_not_zero(E[i], N))
		{
			if (is_in_lp(lpE[N - 1 - i]))
			{
				int k;
#pragma omp for schedule(dynamic,chunk_size)
				for (k = 0; k < N - 3; k += 4)
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
				Reset(E[i], lpE, N - 1 - i);
			}
			else
			{
				if (lpE[N - 1 - i] != -1)
				{
					lp[temp] = lpE[N - i - 1];
					temp++;
					int j;
#pragma omp for schedule(dynamic,chunk_size)
					for (j = 0; j < N - 3; j += 4)
					{
						__m128i rtj = _mm_load_si128((__m128i*)(E[i] + j));
						_mm_store_si128((__m128i*)(R[lp[temp - 1]] + j), rtj);
					}
					for (; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					for (int j = 0; j < N; j++)
						R[lp[temp - 1]][j] = E[i][j];
					goto L2;
				}
			}
		}
	L2:i--;
	}
}

//打印结果矩阵
void print_result()
{
	for (int i = N - 1; i > N - 1 - e; i--)
	{
		for (int j = N - 1; j > -1; j--)
			//cout << E[i][j] << " ";
		{
			if (E[i][j] != 0)
				cout << j << " ";
		}
		cout << endl;
	}
	cout << endl;
}
int main()
{
	for (int i = 0; i < N; i++)
	{
		lp[i] = lpE[i] = -1;
		for (int j = 0; j < N; j++)
			txt_e[i][j] = txt_r[i][j] = -1;
	}
	//处理消元行
	ifstream file("1.txt");
	string line;

	int i = 0;
	while (getline(file, line))
	{
		const char s[2] = " ";
		char* ptr = const_cast<char*>(line.c_str());
		char* token = strtok(ptr, s);
		int j = 0;
		while (token != NULL)
		{
			txt_r[i][j] = atoi(token);
			token = strtok(NULL, s);
			j++;
		}
		i++;
	}
	file.close();

	for (int i = 0; i < r; i++)
	{
		lp[i] = txt_r[i][0];
		for (int j = 0; j < N; j++)
		{
			if (txt_r[i][j] != -1)
				R[txt_r[i][0]][txt_r[i][j]] = 1;
		}
	}

	//处理被消元行
	ifstream file2("2.txt");
	string line2;

	i = 0;
	while (getline(file2, line))
	{
		const char s[2] = " ";
		char* ptr = const_cast<char*>(line.c_str());
		char* token = strtok(ptr, s);
		int j = 0;
		while (token != NULL)
		{
			txt_e[i][j] = atoi(token);
			token = strtok(NULL, s);
			j++;
		}
		i++;
	}
	file2.close();

	for (int i = 0; i < e; i++)
	{
		lpE[i] = txt_e[i][0];
		for (int j = 0; j < N; j++)
		{
			if (txt_e[i][j] != -1)
				E[N - i - 1][txt_e[i][j]] = 1;
		}
	}

	long long head, tail, freq;  //timers
	long long time = 0;

	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Groebner_omp_AVX(R, E, e);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	time = tail - head;
	cout << "Groebner_omp_AVX Time: " << (time * 1000.0 / freq) / 1.0 << "ms" << endl;

	return 0;
}