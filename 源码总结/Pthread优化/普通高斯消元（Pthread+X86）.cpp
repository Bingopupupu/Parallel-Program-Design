#define HAVE_STRUCT_TIMESPEC
#pragma comment(lib, "pthreadVC2.lib")

#include <iostream>
#include <nmmintrin.h> //SSSE4.2
#include <pthread.h>
#include <semaphore.h>
#include <windows.h>
#include <stdlib.h>
#include <iomanip>
#include <immintrin.h> //AVX、AVX2、AVX-512
using namespace std;

//------------------------------------------ 线程控制变量 ------------------------------------------
typedef struct
{
	int k; //消去的轮次
	int t_id; // 线程 id
}threadParam_t_dynamic;

typedef struct
{
	int t_id; //线程编号
} threadParam_t;  //传参打包

//pthread_static2信号量定义
const int NUM_THREAD = 6; //线程数
sem_t sem_main;
sem_t sem_workerstart[NUM_THREAD]; // 每个线程有自己专属的信号量
sem_t sem_workerend[NUM_THREAD];

//pthread3_sem信号量定义
sem_t sem_leader;
sem_t sem_Divsion[NUM_THREAD - 1];
sem_t sem_Elimination[NUM_THREAD - 1];

//barrier 定义
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

// ------------------------------------------ 全局变量 ---------------------------------------------
int N;
int Step[1] = { 5000 };
float** d;
float** matrix;
const int L = 100;
const int LOOP = 1;

void init_d(int x);
void init_matrix();
void serial();
void SSE();
void main_threadFunc_static2();
void main_threadFunc_static3();
void main_threadFunc_static4();
void main_pthread_dynamic();
void main_threadFunc_static2_SSE();
void main_threadFunc_static2_AVX();
void print_matrix();

// ------------------------------------------ 初始化 ------------------------------------------
// 保证每次数据都是一致的
void init_d(int x)
{
	N = x;
	d = new float* [N];
	for (int i = 0; i < N; i++)
	{
		d[i] = new float[N];
	}
	for (int i = 0; i < N; i++)
		for (int j = i; j < N; j++)
			d[i][j] = rand() * 1.0 / RAND_MAX * 100;
	for (int i = 0; i < N - 1; i++)
		for (int j = i + 1; j < N; j++)
			for (int k = 0; k < N; k++)
				d[j][k] += d[i][k];
}

// 初始化matrix，保证数据一致
void init_matrix()
{
	matrix = new float* [N];
	for (int i = 0; i < N; i++)
	{
		matrix[i] = new float[N];
	}
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i][j] = d[i][j];
}

//打印
void print_matrix()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%.2f ", matrix[i][j]);
		}
		printf("\n");
	}
}

// 串行算法
void serial()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}

// SSE并行算法
void SSE()
{
	for (int k = 0; k < N; k++)
	{
		__m128 vt = _mm_set1_ps(matrix[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 4 <= N; j = j + 4)
		{
			__m128 va = _mm_loadu_ps(matrix[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_storeu_ps(matrix[k] + j, va);
		}
		for (; j < N; j++)
		{
			matrix[k][j] == matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			__m128 vaik = _mm_set1_ps(matrix[i][k]);
			int j = k + 1;
			for (j; j + 4 <= N; j = j + 4)
			{
				__m128 vakj = _mm_loadu_ps(matrix[k] + j);
				__m128 vaij = _mm_loadu_ps(matrix[i] + j);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(matrix[i] + j, vaij);
			}
			for (j; j < N; j++)
			{
				matrix[i][j] = matrix[i][j] - matrix[k][j] * matrix[i][k];
			}
			matrix[i][k] = 0;
		}
	}
}

//AVX并行算法
void AVX()
{
	for (int k = 0; k < N; k++)
	{
		__m256 Akk = _mm256_set1_ps(matrix[k][k]);
		int j;
		//考虑对齐操作
		for (j = k + 1; j + 7 < N; j += 8)
		{
			__m256 Akj = _mm256_load_ps(matrix[k] + j);
			Akj = _mm256_div_ps(Akj, Akk);
			_mm256_store_ps(matrix[k] + j, Akj);
		}
		for (; j < N; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			__m256 Aik = _mm256_set1_ps(matrix[i][k]);
			for (j = k + 1; j + 7 < N; j += 8)
			{
				__m256 Akj = _mm256_load_ps(matrix[k] + j);
				__m256 Aij = _mm256_load_ps(matrix[i] + j);
				__m256 AikMulAkj = _mm256_mul_ps(Aik, Akj);
				Aij = _mm256_sub_ps(Aij, AikMulAkj);
				_mm256_store_ps(matrix[i] + j, Aij);
			}
			for (; j < N; j++)
			{
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}

// ====================================== pthread-动态线程版本=====================================
void* threadFunc_dynamic1(void* param)
{
	threadParam_t_dynamic* p = (threadParam_t_dynamic*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号

	for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
	{
		for (int j = k + 1; j < N; ++j)
		{
			matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
		}
		matrix[i][k] = 0;
	}
	pthread_exit(NULL);
	return NULL;
}

void main_pthread_dynamic()
{
	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < N; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1.0;

		//创建工作线程，进行消去操作
		pthread_t handles[NUM_THREAD];// 创建对应的 Handle
		threadParam_t_dynamic param[NUM_THREAD];// 创建对应的线程数据结构
		//分配任务
		for (int t_id = 0; t_id < NUM_THREAD; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < NUM_THREAD; t_id++)
		{
			pthread_create(&handles[t_id], NULL, threadFunc_dynamic1, (void*)(&param[t_id]));
		}
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < NUM_THREAD; t_id++)
		{
			pthread_join(handles[t_id], NULL);
		}
	}
}

// ====================================== pthread-静态线程 + 信号量同步版本======================================
void* threadFunc_static2(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
		{
			//消去
			for (int j = k + 1; j < N; ++j)
			{
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0.0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}

void main_threadFunc_static2() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t handles[NUM_THREAD];// 创建对应的 Handle
	threadParam_t param[NUM_THREAD];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_static2, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < N; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1.0;

		//开始唤醒工作线程
		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务

		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}
}

// ================== pthread-静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数==============================
void* threadFunc_static3(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		// 比信号量更简洁的同步方式是使用 barrier
		if (t_id == 0)
		{
			for (int j = k + 1; j < N; j++)
			{
				matrix[k][j] = matrix[k][j] / matrix[k][k];
			}
			matrix[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREAD - 1; ++i)
			{
				sem_post(&sem_Divsion[i]);
			}
		}

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
		{
			//消去
			for (int j = k + 1; j < N; ++j)
			{
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0.0;
		}

		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREAD - 1; ++i)
			{
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}
			for (int i = 0; i < NUM_THREAD - 1; ++i)
			{
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}

	pthread_exit(NULL);
	return NULL;
}

void main_threadFunc_static3()
{
	//初始化信号量
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < NUM_THREAD - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}

	//创建线程
	pthread_t handles[NUM_THREAD];// 创建对应的 Handle
	threadParam_t param[NUM_THREAD];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_static3, (void*)(&param[t_id]));
	}

	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	// 销毁所有信号量
	sem_destroy(&sem_leader);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_destroy(&sem_Divsion[i]);
		sem_destroy(&sem_Elimination[i]);
	}
}

// ==================================== pthread-静态线程 +barrier 同步==========================================
void* threadFunc_static4(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		if (t_id == 0)
		{
			for (int j = k + 1; j < N; j++)
			{
				matrix[k][j] = matrix[k][j] / matrix[k][k];
			}
			matrix[k][k] = 1.0;
		}

		//第一个同步点
		pthread_barrier_wait(&barrier_Division);

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
		{
			//消去
			for (int j = k + 1; j < N; ++j)
			{
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0.0;
		}

		// 第二个同步点
		pthread_barrier_wait(&barrier_Elimination);
	}
	pthread_exit(NULL);
	return NULL;
}

void main_threadFunc_static4() {
	//初始化 barrier
	pthread_barrier_init(&barrier_Division, NULL, NUM_THREAD);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREAD);
	//创建线程
	pthread_t handles[NUM_THREAD];// 创建对应的 Handle
	threadParam_t param[NUM_THREAD];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_static4, (void*)(&param[t_id]));
	}
	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);
}

// ====================================== pthread-静态线程 + 信号量同步+SSE版本======================================
void* threadFunc_static2_SSE(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
		{
			__m128 vaik = _mm_set1_ps(matrix[i][k]);
			int j = k + 1;
			for (j; j + 4 <= N; j = j + 4)
			{
				__m128 vakj = _mm_loadu_ps(matrix[k] + j);
				__m128 vaij = _mm_loadu_ps(matrix[i] + j);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(matrix[i] + j, vaij);
			}
			for (j; j < N; j++)
			{
				matrix[i][j] = matrix[i][j] - matrix[k][j] * matrix[i][k];
			}
			matrix[i][k] = 0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}

void main_threadFunc_static2_SSE() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t handles[NUM_THREAD];// 创建对应的 Handle
	threadParam_t param[NUM_THREAD];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_static2_SSE, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		__m128 vt = _mm_set1_ps(matrix[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 4 <= N; j = j + 4)
		{
			__m128 va = _mm_loadu_ps(matrix[k] + j);
			va = _mm_div_ps(va, vt);
			_mm_storeu_ps(matrix[k] + j, va);
		}
		for (; j < N; j++)
		{
			matrix[k][j] == matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1.0;
		//开始唤醒工作线程
		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务

		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}
}

// ====================================== pthread-静态线程 + 信号量同步+AVX版本======================================
void* threadFunc_static2_AVX(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
		{
			__m256 vaik = _mm256_set1_ps(matrix[i][k]);
			int j = k + 1;
			for (j; j + 4 <= N; j = j + 4)
			{
				__m256 vakj = _mm256_loadu_ps(matrix[k] + j);
				__m256 vaij = _mm256_loadu_ps(matrix[i] + j);
				__m256 vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(matrix[i] + j, vaij);
			}
			for (j; j < N; j++)
			{
				matrix[i][j] = matrix[i][j] - matrix[k][j] * matrix[i][k];
			}
			matrix[i][k] = 0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return NULL;
}

void main_threadFunc_static2_AVX() {
	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t handles[NUM_THREAD];// 创建对应的 Handle
	threadParam_t param[NUM_THREAD];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc_static2_SSE, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		__m256 vt = _mm256_set1_ps(matrix[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 4 <= N; j = j + 4)
		{
			__m256 va = _mm256_loadu_ps(matrix[k] + j);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(matrix[k] + j, va);
		}
		for (; j < N; j++)
		{
			matrix[k][j] == matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1.0;
		//开始唤醒工作线程
		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务

		for (int t_id = 0; t_id < NUM_THREAD; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}
	}

	for (int t_id = 0; t_id < NUM_THREAD; t_id++)
	{
		pthread_join(handles[t_id], NULL);
	}

	//销毁所有信号量
	sem_destroy(&sem_main);
	for (int i = 0; i < NUM_THREAD; ++i)
	{
		sem_destroy(&sem_workerstart[i]);
		sem_destroy(&sem_workerend[i]);
	}
}

int main()
{
	//设立循环次数
	for (int i = 0; i < LOOP; i++)
	{
		long long head, tail, freq;

		//计时初始化
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		float time = 0;
		init_d(Step[i]);
		cout << "N: " << N << endl;
		// ====================================== serial ======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		serial();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "serial:" << time << "ms" << endl;
		// ====================================== SSE ======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "SSE:" << time << "ms" << endl;
		// ====================================== AVX ======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "AVX:" << time << "ms" << endl;
		// ====================================== pthread-动态线程版本======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		main_pthread_dynamic();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "pthread_dynamic:" << time << "ms" << endl;
		// ================================pthread-静态线程 + 信号量同步版本 ======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		main_threadFunc_static2();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "threadFunc_static2:" << time << "ms" << endl;
		// =======================pthread-静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数=====================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		main_threadFunc_static3();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "threadFunc_static3:" << time << "ms" << endl;
		// ================================pthread-静态线程 +barrier 同步======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		main_threadFunc_static4();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "threadFunc_static4:" << time << "ms" << endl;
		// ====================================== pthread-两重 信号量 SSE ======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		main_threadFunc_static2_SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "threadFunc_static2_SSE:" << time << "ms" << endl;
		// ====================================== pthread-两重 信号量 AVX ======================================
		time = 0;
		init_matrix();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		main_threadFunc_static2_AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		time = (tail - head) * 1000.0 / freq;
		cout << "threadFunc_static2_AVX:" << time << "ms" << endl;
	}
	return 0;
}