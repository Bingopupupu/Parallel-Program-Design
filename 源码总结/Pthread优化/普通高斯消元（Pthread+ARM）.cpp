#include <iostream>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
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
const int NUM_THREAD = 8; //线程数
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
int Step[6] = { 500,1000,1500,2000,2500,3000 };
float** d;
float** matrix;
const int L = 100;
const int LOOP = 1;

void init_d(int x);
void init_matrix();
void serial();
void main_threadFunc_static2();
void main_threadFunc_static3();
void main_threadFunc_static4();
void main_pthread_dynamic();
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

void Neon()
{
	for (int k = 0; k < N; k++)
	{
		float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
		int j;
		for (j = k + 1; j + 3 < N; j += 4)
		{
			float32x4_t Akj = vld1q_f32(matrix[k] + j);
			Akj = vdivq_f32(Akj, Akk);
			vst1q_f32(matrix[k] + j, Akj);
		}
		for (; j < N; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
			for (j = k + 1; j + 3 < N; j += 4)
			{
				float32x4_t Akj = vld1q_f32(matrix[k] + j);
				float32x4_t Aij = vld1q_f32(matrix[i] + j);
				float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
				Aij = vsubq_f32(Aij, AikMulAkj);
				vst1q_f32(matrix[i] + j, Aij);
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

// ====================================== pthread-静态线程 + 信号量同步+Neon版本======================================
void* threadFunc2_static2_neon(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += NUM_THREAD)
		{
			float32x4_t vaik = vmovq_n_f32(matrix[i][k]);
			int j = k + 1;
			for (j; j + 4 <= N; j = j + 4)
			{
				float32x4_t vakj = vld1q_f32(matrix[k] + j);
				float32x4_t vaij = vld1q_f32(matrix[i] + j);
				float32x4_t vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(matrix[i] + j, vaij);
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

void main_threadFunc_static2_neon() {
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
		pthread_create(&handles[t_id], NULL, threadFunc2_static2_neon, (void*)(&param[t_id]));
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做NEON除法操作
		float32x4_t vt = vmovq_n_f32(matrix[k][k]);
		int j = k + 1;
		for (j = k + 1; j + 4 <= N; j = j + 4)
		{
			float32x4_t va = vld1q_f32(matrix[k] + j);
			va = vdivq_f32(va, vt);
			vst1q_f32(matrix[k] + j, va);
		}
		for (j; j < N; j++)
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

int main()
{
	struct timeval start;
	struct timeval end;
	float time = 0;
	for (int ii = 0; ii < 6; ii++)
	{

		init_d(Step[ii]);
		cout << N << endl;
		// ====================================== serial ======================================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			//print_matrix();
			gettimeofday(&start, NULL);
			serial();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "serial:" << time / LOOP << "ms" << endl;
		//print_matrix();

		// ====================================== neon ======================================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			gettimeofday(&start, NULL);
			Neon();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "neon:" << time / LOOP << "ms" << endl;
		//print_matrix();
		
		// ====================================== pthread-动态线程 ======================================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			gettimeofday(&start, NULL);
			main_pthread_dynamic();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "pthread_dynamic:" << time / LOOP << "ms" << endl;

		// ================================pthread-静态线程 + 信号量同步版本 ======================================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			gettimeofday(&start, NULL);
			main_threadFunc_static2();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "threadFunc_static2:" << time / LOOP << "ms" << endl;
		//print_matrix();


		// ===========================pthread-静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数========================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			gettimeofday(&start, NULL);
			main_threadFunc_static3();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "threadFunc_static3:" << time / LOOP << "ms" << endl;
		//print_matrix();

		// ================================pthread-静态线程 +barrier 同步==================================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			gettimeofday(&start, NULL);
			main_threadFunc_static4();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "threadFunc_static4:" << time / LOOP << "ms" << endl;
		//print_matrix();

		// =============================== pthread-静态线程 + 信号量同步+NEON版本====================================
		time = 0;
		for (int i = 0; i < LOOP; i++)
		{
			init_matrix();
			gettimeofday(&start, NULL);
			main_threadFunc_static2_neon();
			gettimeofday(&end, NULL);
			time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
		}
		cout << "main_threadFunc_static2_neon:" << time / LOOP << "ms" << endl;

		for (int i = 0; i < N; i++)
		{
			delete d[i];
		}
	}
	delete d;
	return 0;
}