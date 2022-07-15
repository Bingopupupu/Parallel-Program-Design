#include <iostream>
#include <windows.h>

using namespace std;
//使用SSE intrinsics所需的头文件
#include <xmmintrin.h>  //SSE
#include <emmintrin.h>  //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h>  //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h>  //SSSE4.2
#include <immintrin.h> //AVX、AVX2、AVX-512

//使用Neon intrinsics所需的头文件
//#include <arm_neon.h>

const int N = 128;

//初始化矩阵
void init(float A[N][N], int n, int value)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i][j] = value + 1;
			value++;
		}
	}
}

//打印矩阵
void printMatrix(float A[N][N], int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << A[i][j] << " ";
		}
		cout << endl;
	}
}

//C++串行算法
void GaussElimination_Serial_Cpp(float A[N][N], int n)
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

//Neon指令高斯消元
//void GaussElimination_SIMD_Neon(float A[N][N], int n)
//{
//    for (int k = 0; k < n; k++)
//    {
//        float32x4_t Akk = vmovq_n_f32(A[k][k]);
//        int j;
//        for (j = k + 1; j + 3 < n; j += 4)
//        {
//            float32x4_t Akj = vld1q_f32(A[k] + j);
//            Akj = vdivq_f32(Akj, Akk);
//            vst1q_f32(A[k] + j, Akj);
//        }
//        for (; j < n; j++)
//        {
//            A[k][j] = A[k][j] / A[k][k];
//        }
//        A[k][k] = 1;
//        for (int i = k + 1; i < n; i++)
//        {
//            float32x4_t Aik = vmovq_n_f32(A[i][k]);
//            for (j = k + 1; j + 3 < n; j += 4)
//            {
//                float32x4_t Akj = vld1q_f32(A[k] + j);
//                float32x4_t Aij = vld1q_f32(A[i] + j);
//                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
//                Aij = vsubq_f32(Aij, AikMulAkj);
//                vst1q_f32(A[i] + j, Aij);
//            }
//            for (; j < n; j++)
//            {
//                A[i][j] = A[i][j] - A[i][k] * A[k][j];
//            }
//            A[i][k] = 0;
//        }
//    }
//}

//SSE并行算法
void GaussElimination_SIMD_SSE(float A[N][N], int n)
{
	for (int k = 0; k < n; k++)
	{
		// float Akk = A[k][k];
		__m128 Akk = _mm_set_ps1(A[k][k]);
		int j;
		//考虑对齐操作
		for (j = k + 1; j + 3 < n; j += 4)
		{
			//float Akj = A[k][j];
			__m128 Akj = _mm_load_ps(A[k] + j);
			// Akj = Akj / Akk;
			Akj = _mm_div_ps(Akj, Akk);
			//Akj = A[k][j];
			_mm_store_ps(A[k] + j, Akj);
		}
		for (; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			// float Aik = A[i][k];
			__m128 Aik = _mm_set_ps1(A[i][k]);
			for (j = k + 1; j + 3 < n; j += 4)
			{
				//float Akj = A[k][j];
				__m128 Akj = _mm_load_ps(A[k] + j);
				//float Aij = A[i][j];
				__m128 Aij = _mm_load_ps(A[i] + j);
				// AikMulAkj = A[i][k] * A[k][j];
				__m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
				// Aij = Aij - AikMulAkj;
				Aij = _mm_sub_ps(Aij, AikMulAkj);
				//A[i][j] = Aij;
				_mm_store_ps(A[i] + j, Aij);
			}
			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

//AVX并行算法
void GaussElimination_SIMD_AVX(float A[N][N], int n)
{
	for (int k = 0; k < n; k++)
	{
		// float Akk = A[k][k];
		__m256 Akk = _mm256_set1_ps(A[k][k]);
		int j;
		//考虑对齐操作
		for (j = k + 1; j + 7 < n; j += 8)
		{
			//float Akj = A[k][j];
			__m256 Akj = _mm256_load_ps(A[k] + j);
			// Akj = Akj / Akk;
			Akj = _mm256_div_ps(Akj, Akk);
			//Akj = A[k][j];
			_mm256_store_ps(A[k] + j, Akj);
		}
		for (; j < n; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1;
		for (int i = k + 1; i < n; i++)
		{
			// float Aik = A[i][k];
			__m256 Aik = _mm256_set1_ps(A[i][k]);
			for (j = k + 1; j + 7 < n; j += 8)
			{
				//float Akj = A[k][j];
				__m256 Akj = _mm256_load_ps(A[k] + j);
				//float Aij = A[i][j];
				__m256 Aij = _mm256_load_ps(A[i] + j);
				// AikMulAkj = A[i][k] * A[k][j];
				__m256 AikMulAkj = _mm256_mul_ps(Aik, Akj);
				// Aij = Aij - AikMulAkj;
				Aij = _mm256_sub_ps(Aij, AikMulAkj);
				//A[i][j] = Aij;
				_mm256_store_ps(A[i] + j, Aij);
			}
			for (; j < n; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

int main()
{
	float A[N][N]; int value = 0;

	long long head, tail, freq; // timers
	QueryPerformanceFrequency((LARGE_INTEGER *) & freq);

	// start time
	QueryPerformanceCounter((LARGE_INTEGER *) & head);
	for (int i = 0; i < 100; i++)
	{

		init(A, N, 0);
		GaussElimination_Serial_Cpp(A, N);

		//init(A, N, 0);
		//GaussElimination_SIMD_SSE(A, N);

		//init(A, N, 0);
		//GaussElimination_SIMD_AVX(A, N);
	}
	QueryPerformanceCounter((LARGE_INTEGER *) & tail);
	cout <<"Col: "<< (tail - head) * 1000.0 / freq << "ms" << endl;

	return 0;
}