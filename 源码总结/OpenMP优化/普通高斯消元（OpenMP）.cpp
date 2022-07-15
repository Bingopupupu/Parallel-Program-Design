#include<iostream>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<immintrin.h>
#include<tmmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include <nmmintrin.h> 
#include <omp.h>
//#include<arm_neon.h>

using namespace std;

//全局变量
int N;
int step[5] = { 50,100 ,500,1000 ,5000 };
//设置地址对齐策略
__declspec(align(16))float** Gauss;//待消元的矩阵
__declspec(align(16))float** UP;//上三角矩阵
//线程数
const int thread_count = 12;
const int chunk_size = 24;


//N阶上三角阵（每个元素均为1）
void generate_up(int n)
{
	N = n;
	UP = new float* [N];
	for (int i = 0; i < N; i++)
		UP[i] = new float[N];
	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			UP[i][j] = 1;
		}
	}
}

//生成待高斯消元的矩阵
void generate_gauss()
{
	Gauss = new float* [N];
	for (int i = 0; i < N; i++)
		Gauss[i] = new float[N];
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			Gauss[i][j] = UP[i][j];
	}
	for (int i = 0; i < N - 1; i++)
	{
		for (int j = 0; j < N; j++)
			Gauss[i + 1][j] = Gauss[i][j] + Gauss[i + 1][j];
	}
}

//判断
bool is_right()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (Gauss[i][j] != UP[i][j])
				return false;
		}
	}
	return true;
}

//串行算法:普通高斯消元
void Serial()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//SSE指令高斯消元
void SSE()
{
	for (int k = 0; k < N; k++)
	{
		__m128 vt = _mm_set_ps1(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			__m128 va = _mm_load_ps(Gauss[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			__m128 vaik = _mm_set_ps1(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				__m128 vakj = _mm_load_ps(Gauss[k] + j);
				__m128 vaij = _mm_load_ps(Gauss[i] + j);
				__m128 vx = _mm_mul_ps(vaik, vakj);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}

//AVX指令高斯消元
void AVX()
{
	for (int k = 0; k < N; k++)
	{
		__m256 vt = _mm256_set1_ps(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 7 < N; j += 8)
		{
			__m256 va = _mm256_load_ps(Gauss[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(Gauss[k] + j, va);
		}
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			__m256 vaik = _mm256_set1_ps(Gauss[i][k]);
			for (j = k + 1; j + 7 < N; j += 8)
			{
				__m256 vakj = _mm256_load_ps(Gauss[k] + j);
				__m256 vaij = _mm256_load_ps(Gauss[i] + j);
				__m256 vx = _mm256_mul_ps(vaik, vakj);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}

//omp_static版本高斯消元
void OpenMP_static()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
#pragma omp for schedule(static)
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//omp_dynamic版本高斯消元
void OpenMp_dynamic()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
#pragma omp for schedule(dynamic,chunk_size)
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(dynamic,chunk_size)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//omp_guided版本高斯消元
void OpenMP_guided()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
#pragma omp for schedule(guided,chunk_size)
		for (int j = k + 1; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(guided,chunk_size)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
}

//omp_SSE版本高斯消元
void OpenMP_SSE_dynamic()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
		__m128 vt = _mm_set_ps1(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			__m128 va = _mm_load_ps(Gauss[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(Gauss[k] + j, va);
		}
#pragma omp single
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(dynamic, chunk_size)
		for (int i = k + 1; i < N; i++)
		{
			__m128 vaik = _mm_set_ps1(Gauss[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				__m128 vakj = _mm_load_ps(Gauss[k] + j);
				__m128 vaij = _mm_load_ps(Gauss[i] + j);
				__m128 vx = _mm_mul_ps(vaik, vakj);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}

//omp_AVX版本高斯消元
void OpenMP_AVX_dynamic()
{
#pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
		__m256 vt = _mm256_set1_ps(Gauss[k][k]);
		int j;
		for (j = k + 1; j + 7 < N; j += 8)
		{
			__m256 va = _mm256_load_ps(Gauss[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(Gauss[k] + j, va);
		}
#pragma omp single
		for (; j < N; j++)
		{
			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
		}
		Gauss[k][k] = 1;
#pragma omp for schedule(dynamic, 24)
		for (int i = k + 1; i < N; i++)
		{
			__m256 vaik = _mm256_set1_ps(Gauss[i][k]);
			for (j = k + 1; j + 7 < N; j += 8)
			{
				__m256 vakj = _mm256_load_ps(Gauss[k] + j);
				__m256 vaij = _mm256_load_ps(Gauss[i] + j);
				__m256 vx = _mm256_mul_ps(vaik, vakj);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(Gauss[i] + j, vaij);
			}
			for (; j < N; j++)
			{
				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
			}
			Gauss[i][k] = 0;
		}
	}
	//print(Gauss, N);
}

int main()
{
	for (int i = 0; i < 5; i++)
	{
		long long head, tail, freq;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		float time = 0;
		generate_up(step[i]);
		cout << "N: " << N << endl;

		//串行高斯消元
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Serial();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "串行高斯消元:" << time << "ms" << endl;

		//SSE版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "SSE版本高斯消元:" << time << "ms" << endl;

		//AVX版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "AVX版本高斯消元:" << time << "ms" << endl;

		//omp_static版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		OpenMP_static();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_static版本高斯消元:" << time << "ms" << endl;

		//omp_dynamic版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		OpenMp_dynamic();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_dynamic版本高斯消元:" << time << "ms" << endl;

		//omp_guided版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		OpenMP_guided();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_guided版本高斯消元:" << time << "ms" << endl;

		//omp_SSE版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		OpenMP_SSE_dynamic();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_SSE版本高斯消元:" << time << "ms" << endl;

		//omp_AVX版本高斯消元
		time = 0;
		generate_gauss();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		OpenMP_AVX_dynamic();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "omp_AVX版本高斯消元:" << time << "ms" << endl;

		cout << endl;
	}
}












//ARM平台
//#include<iostream>
//#include <sys/time.h>
//#include <omp.h>
//#include<arm_neon.h>
//
//using namespace std;
//
////全局变量
//int N;
//int step[5] = { 16,64,256,1024,4096 };
//float** Gauss;//待消元的矩阵
//float** UP;//上三角矩阵
////线程数
//const int thread_count = 4;
//
////处理函数
//
////N阶上三角阵（每个元素均为1）
//void generate_up(int n)
//{
//	N = n;
//	UP = new float* [N];
//	for (int i = 0; i < N; i++)
//		UP[i] = new float[N];
//	for (int i = 0; i < N; i++)
//	{
//		for (int j = i; j < N; j++)
//		{
//			UP[i][j] = 1;
//		}
//	}
//	//print(UP, n);
//}
//
////生成待高斯消元的矩阵
//void generate_gauss()
//{
//	Gauss = new float* [N];
//	for (int i = 0; i < N; i++)
//		Gauss[i] = new float[N];
//	for (int i = 0; i < N; i++)
//	{
//		for (int j = 0; j < N; j++)
//			Gauss[i][j] = UP[i][j];
//	}
//	for (int i = 0; i < N - 1; i++)
//	{
//		for (int j = 0; j < N; j++)
//			Gauss[i + 1][j] = Gauss[i][j] + Gauss[i + 1][j];
//	}
//	//print(A, n);
//}
//
////判断
//bool is_right()
//{
//	for (int i = 0; i < N; i++)
//	{
//		for (int j = 0; j < N; j++)
//		{
//			if (Gauss[i][j] != UP[i][j])
//				return false;
//		}
//	}
//	return true;
//}
//
////打印
//void print(float** A, int n)
//{
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			cout << A[i][j] << ",";
//		cout << endl;
//	}
//}
//
////串行算法:普通高斯消元
//void Gauss_ordinary()
//{
//	for (int k = 0; k < N; k++)
//	{
//		for (int j = k + 1; j < N; j++)
//		{
//			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
//		}
//		Gauss[k][k] = 1;
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
//			}
//			Gauss[i][k] = 0;
//		}
//	}
//}
//
////Neon指令高斯消元
//void Gauss_SIMD_Neon()
//{
//	for (int k = 0; k < N; k++)
//	{
//		float32x4_t vt = vmovq_n_f32(Gauss[k][k]);
//		int j;
//		for (j = k + 1; j + 3 < N; j += 4)
//		{
//			float32x4_t va = vld1q_f32(Gauss[k] + j);
//			va = vdivq_f32(va, vt);
//			vst1q_f32(Gauss[k] + j, va);
//		}
//		for (; j < N; j++)
//		{
//			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
//		}
//		Gauss[k][k] = 1;
//		for (int i = k + 1; i < N; i++)
//		{
//			float32x4_t vaik = vmovq_n_f32(Gauss[i][k]);
//			for (j = k + 1; j + 3 < N; j += 4)
//			{
//				float32x4_t vakj = vld1q_f32(Gauss[k] + j);
//				float32x4_t vaij = vld1q_f32(Gauss[i] + j);
//				float32x4_t vx = vmulq_f32(vaik, vakj);
//				vaij = vsubq_f32(vaij, vx);
//				vst1q_f32(Gauss[i] + j, vaij);
//			}
//			for (; j < N; j++)
//			{
//				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
//			}
//			Gauss[i][k] = 0;
//		}
//	}
//	//print(A, N);
//}
//
////omp_static版本高斯消元
//void Gauss_omp_static()
//{
//#pragma omp parallel num_threads(thread_count)
//	for (int k = 0; k < N; k++)
//	{
//#pragma omp for schedule(static)
//		for (int j = k + 1; j < N; j++)
//		{
//			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
//		}
//		Gauss[k][k] = 1;
//#pragma omp for schedule(static)
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
//			}
//			Gauss[i][k] = 0;
//		}
//	}
//}
//
////omp_dynamic版本高斯消元
//void Gauss_omp_dynamic()
//{
//#pragma omp parallel num_threads(thread_count)
//	for (int k = 0; k < N; k++)
//	{
//#pragma omp for schedule(dynamic,24)
//		for (int j = k + 1; j < N; j++)
//		{
//			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
//		}
//		Gauss[k][k] = 1;
//#pragma omp for schedule(dynamic,24)
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
//			}
//			Gauss[i][k] = 0;
//		}
//	}
//}
//
////omp_guided版本高斯消元
//void Gauss_omp_guided()
//{
//#pragma omp parallel num_threads(thread_count)
//	for (int k = 0; k < N; k++)
//	{
//#pragma omp for schedule(guided,24)
//		for (int j = k + 1; j < N; j++)
//		{
//			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
//		}
//		Gauss[k][k] = 1;
//#pragma omp for schedule(guided,24)
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
//			}
//			Gauss[i][k] = 0;
//		}
//	}
//}
//
////omp_Neon指令高斯消元
//void Gauss_Neon_omp()
//{
//#pragma omp parallel num_threads(thread_count)
//	for (int k = 0; k < N; k++)
//	{
//		float32x4_t vt = vmovq_n_f32(Gauss[k][k]);
//		int j;
//		for (j = k + 1; j + 3 < N; j += 4)
//		{
//			float32x4_t va = vld1q_f32(Gauss[k] + j);
//			va = vdivq_f32(va, vt);
//			vst1q_f32(Gauss[k] + j, va);
//		}
//#pragma omp single
//		for (; j < N; j++)
//		{
//			Gauss[k][j] = Gauss[k][j] / Gauss[k][k];
//		}
//		Gauss[k][k] = 1;
//#pragma omp for schedule(dynamic, 24)
//		for (int i = k + 1; i < N; i++)
//		{
//			float32x4_t vaik = vmovq_n_f32(Gauss[i][k]);
//			for (j = k + 1; j + 3 < N; j += 4)
//			{
//				float32x4_t vakj = vld1q_f32(Gauss[k] + j);
//				float32x4_t vaij = vld1q_f32(Gauss[i] + j);
//				float32x4_t vx = vmulq_f32(vaik, vakj);
//				vaij = vsubq_f32(vaij, vx);
//				vst1q_f32(Gauss[i] + j, vaij);
//			}
//			for (; j < N; j++)
//			{
//				Gauss[i][j] = Gauss[i][j] - Gauss[i][k] * Gauss[k][j];
//			}
//			Gauss[i][k] = 0;
//		}
//	}
//	//print(A, N);
//}
//
//int main()
//{
//	for (int i = 0; i < 5; i++)
//	{
//		struct timeval start;
//		struct timeval end;
//		float time = 0;
//		generate_up(step[i]);
//		cout << "N: " << N << endl;
//
//		//串行高斯消元
//		generate_gauss();
//		gettimeofday(&start, NULL);
//		Gauss_ordinary();
//		gettimeofday(&end, NULL);
//		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
//		cout << "串行高斯消元:" << time << "ms" << endl;
//
//
//		//omp_static版本高斯消元
//		time = 0;
//		generate_gauss();
//		gettimeofday(&start, NULL);
//		Gauss_omp_static();
//		gettimeofday(&end, NULL);
//		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
//		cout << "omp_static版本高斯消元:" << time << "ms" << endl;
//
//		//omp_dynamic版本高斯消元
//		time = 0;
//		generate_gauss();
//		gettimeofday(&start, NULL);
//		Gauss_omp_dynamic();
//		gettimeofday(&end, NULL);
//		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
//		cout << "omp_dynamic版本高斯消元:" << time << "ms" << endl;
//
//		//omp_guided版本高斯消元
//		time = 0;
//		generate_gauss();
//		gettimeofday(&start, NULL);
//		Gauss_omp_guided();
//		gettimeofday(&end, NULL);
//		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
//		cout << "omp_guided版本高斯消元:" << time << "ms" << endl;
//
//		//omp_Neon版本高斯消元
//		time = 0;
//		generate_gauss();
//		gettimeofday(&start, NULL);
//		Gauss_Neon_omp();
//		gettimeofday(&end, NULL);
//		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
//		cout << "omp_Neon版本高斯消元:" << time << "ms" << endl;
//
//		//Neon版本高斯消元
//		time = 0;
//		generate_gauss();
//		gettimeofday(&start, NULL);
//		Gauss_SIMD_Neon();
//		gettimeofday(&end, NULL);
//		time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
//		cout << "Neon版本高斯消元:" << time << "ms" << endl;
//
//		cout << endl;
//	}
//}