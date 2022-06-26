#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>

#define Satellite 1
#define Planet 2
#define Black_hole 3
#define Asteroid 4
#define Star 5
#define Device 6
#define NumOfThreads 128 

__constant__ int dn_steps = 200000;
__constant__ double ddt = 60;
__constant__ double deps = 1e-3;
__constant__ double dG = 6.674e-11;
__constant__ double dplanet_radius = 1e7;
__constant__ double dmissile_speed = 1e6;
__device__ double dget_missile_cost(double t) { return 1e5 + 1e3 * t; }
__device__ double dgravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

double min_dist2;

namespace param {
    const int n_steps = 200000;
    const double dt = 60;
    const double eps = 1e-3;
    const double G = 6.674e-11;
    double gravity_device_mass(double m0, double t) {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000));
    }
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_basic(char* filename, int& n, int& planet, int& asteroid) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
}

void read_input(char* filename, int& n, double4* q, double4 *v) {
    std::ifstream fin(filename);
    int a;
    fin >> a >> a >> a;
    for (int i = 0; i < n; i++) {
        std::string str;
        fin >> q[i].x >> q[i].y >> q[i].z >> v[i].x >> v[i].y >> v[i].z >> q[i].w >> str;
        if (str == "satellite") v[i].w = Satellite;
        else if (str == "planet") v[i].w = Planet;
        else if (str == "black_hole") v[i].w = Black_hole;
        else if (str == "asteroid") v[i].w = Asteroid;
        else if (str == "star") v[i].w = Star;
        else if (str == "device") v[i].w = Device;
        else v[i].w = Satellite;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

__global__ void run_step(int step, int n, double4* q, double4* v) {
    if (blockIdx.x >= n || threadIdx.x >= n) return ;

    // compute accelerations
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    double4 qj, qi;
    double mj;
    double4 a;
    a.x = a.y = a.z = 0;

    __shared__ double4 r[NumOfThreads];

    qi.x = q[bid].x;
    qi.y = q[bid].y;
    qi.z = q[bid].z;

    for (int i = tid; i < n; i += blockDim.x) {
        // qj = q[i];
        qj.x = q[i].x;
        qj.y = q[i].y;
        qj.z = q[i].z;

        mj = q[i].w;
        if (v[i].w == Device) {
            mj = dgravity_device_mass(mj, step * ddt);
        }
        double dx = qj.x - qi.x;
        double dy = qj.y - qi.y;
        double dz = qj.z - qi.z;
        double dist3 =
            sqrt((dx * dx + dy * dy + dz * dz + param::eps * param::eps) * (dx * dx + dy * dy + dz * dz + param::eps * param::eps) * (dx * dx + dy * dy + dz * dz + param::eps * param::eps));

        a.x += dG * mj * dx / dist3;
        a.y += dG * mj * dy / dist3;
        a.z += dG * mj * dz / dist3;
    }

    r[tid] = a;
    __syncthreads();

    for (int size = NumOfThreads / 2; size > 0; size /= 2) { //uniform
        if (tid < size) {
            r[tid].x += r[tid + size].x;
            r[tid].y += r[tid + size].y;
            r[tid].z += r[tid + size].z;
        }
        __syncthreads();
    }
    if (tid == 0) {
        v[bid].x += r[0].x * ddt;
        v[bid].y += r[0].y * ddt;
        v[bid].z += r[0].z * ddt;
    }
}

__global__ void run_step2(int step, int n, double4* q, double4* v, int *hit_time_step) {
    if (*hit_time_step != -2) return ;
    if (blockIdx.x >= n || threadIdx.x >= n) return ;

    // compute accelerations
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    double4 qj, qi;
    double mj;
    double4 a;

    a.x = a.y = a.z = 0;

    __shared__ double4 r[NumOfThreads];

    qi.x = q[bid].x;
    qi.y = q[bid].y;
    qi.z = q[bid].z;

    for (int i = tid; i < n; i += blockDim.x) {
        qj.x = q[i].x;
        qj.y = q[i].y;
        qj.z = q[i].z;

        mj = q[i].w;
        if (v[i].w == Device) {
            mj = dgravity_device_mass(mj, step * ddt);
        }
        double dx = qj.x - qi.x;
        double dy = qj.y - qi.y;
        double dz = qj.z - qi.z;
        double dist3 =
            sqrt((dx * dx + dy * dy + dz * dz + param::eps * param::eps) * (dx * dx + dy * dy + dz * dz + param::eps * param::eps) * (dx * dx + dy * dy + dz * dz + param::eps * param::eps));

        a.x += dG * mj * dx / dist3;
        a.y += dG * mj * dy / dist3;
        a.z += dG * mj * dz / dist3;
    }

    r[tid] = a;

    __syncthreads();

    for (int size = NumOfThreads / 2; size > 0; size /= 2) { //uniform
        if (tid < size) {
            r[tid].x += r[tid + size].x;
            r[tid].y += r[tid + size].y;
            r[tid].z += r[tid + size].z;
        }
        __syncthreads();
    }
    if (tid == 0) {
        v[bid].x += r[0].x * ddt;
        v[bid].y += r[0].y * ddt;
        v[bid].z += r[0].z * ddt;
    }
}

__global__ void run_step3(int step, int n, double4* q, double4* v, int *isHit) {
    if (*isHit == 1) return ;
    if (blockIdx.x >= n || threadIdx.x >= n) return ;

    // compute accelerations
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    double4 qj, qi;
    double mj;
    double4 a;

    a.x = a.y = a.z = 0;

    __shared__ double4 r[NumOfThreads];

    qi.x = q[bid].x;
    qi.y = q[bid].y;
    qi.z = q[bid].z;

    for (int i = tid; i < n; i += blockDim.x) {
        // qj = q[i];
        qj.x = q[i].x;
        qj.y = q[i].y;
        qj.z = q[i].z;

        mj = q[i].w;
        if (v[i].w == Device) {
            mj = dgravity_device_mass(mj, step * ddt);
        }
        double dx = qj.x - qi.x;
        double dy = qj.y - qi.y;
        double dz = qj.z - qi.z;
        double dist3 =
            sqrt((dx * dx + dy * dy + dz * dz + param::eps * param::eps) * (dx * dx + dy * dy + dz * dz + param::eps * param::eps) * (dx * dx + dy * dy + dz * dz + param::eps * param::eps));

        a.x += dG * mj * dx / dist3;
        a.y += dG * mj * dy / dist3;
        a.z += dG * mj * dz / dist3;
    }

    r[tid] = a;

    __syncthreads();

    for (int size = NumOfThreads / 2; size > 0; size /= 2) { //uniform
        if (tid < size) {
            r[tid].x += r[tid + size].x;
            r[tid].y += r[tid + size].y;
            r[tid].z += r[tid + size].z;
        }
        __syncthreads();
    }
    if (tid == 0) {
        v[bid].x += r[0].x * ddt;
        v[bid].y += r[0].y * ddt;
        v[bid].z += r[0].z * ddt;
    }
}

__global__ void update_position(int n, double4* q, double4* v, double *min_dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || idx == 1) return ;

    q[idx].x += v[idx].x * ddt;
    q[idx].y += v[idx].y * ddt;
    q[idx].z += v[idx].z * ddt;

    if (idx == 0) {
        q[1].x += v[1].x * ddt;
        q[1].y += v[1].y * ddt;
        q[1].z += v[1].z * ddt;

        double4 d;
        d.x = q[0].x - q[1].x;
        d.y = q[0].y - q[1].y;
        d.z = q[0].z - q[1].z;
        double dist = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (*min_dist > dist) {
            *min_dist = dist;
        }
    }
}

__global__ void update_position2(int n, int step, double4* q, double4* v, int *hit_time_step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || idx == 1) return ;
    if (*hit_time_step != -2) return ;

    q[idx].x += v[idx].x * ddt;
    q[idx].y += v[idx].y * ddt;
    q[idx].z += v[idx].z * ddt;

    if (idx == 0) {
        q[1].x += v[1].x * ddt;
        q[1].y += v[1].y * ddt;
        q[1].z += v[1].z * ddt;

        double4 d;
        d.x = q[0].x - q[1].x;
        d.y = q[0].y - q[1].y;
        d.z = q[0].z - q[1].z;
        double dist = d.x * d.x + d.y * d.y + d.z * d.z;
        if (dist < dplanet_radius * dplanet_radius) {
            *hit_time_step = step;
        }
    }
}

__global__ void update_position3(int n, int step, double4* q, double4* v, int *isHit, double *cost) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || idx == 1 || idx == 2) return ;
    if (*isHit == 1) return ;

    q[idx].x += v[idx].x * ddt;
    q[idx].y += v[idx].y * ddt;
    q[idx].z += v[idx].z * ddt;

    if (idx == 0) {
        q[1].x += v[1].x * ddt;
        q[1].y += v[1].y * ddt;
        q[1].z += v[1].z * ddt;

        q[2].x += v[2].x * ddt;
        q[2].y += v[2].y * ddt;
        q[2].z += v[2].z * ddt;

        double4 d;
        double dist;
        d.x = q[0].x - q[2].x;
        d.y = q[0].y - q[2].y;
        d.z = q[0].z - q[2].z;

        dist = d.x * d.x + d.y * d.y + d.z * d.z;
        if (*cost == 0) {
            if (dist < (step) * ddt * dmissile_speed * (step) * ddt * dmissile_speed) {
                q[2].w = 0;
                *cost = dget_missile_cost((step + 1) * ddt);
            }
        }

        d.x = q[0].x - q[1].x;
        d.y = q[0].y - q[1].y;
        d.z = q[0].z - q[1].z;
        dist = d.x * d.x + d.y * d.y + d.z * d.z;
        if (dist < dplanet_radius * dplanet_radius) {
            *isHit = 1;
        }
    }
}

__global__ void update_position_sample(int n, double4* q, double4* v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return ;

    q[idx].x += v[idx].x * ddt;
    q[idx].y += v[idx].y * ddt;
    q[idx].z += v[idx].z * ddt;
}

void problem1(char *filename, int n, int planet, int asteroid, std::vector<int> todo, double &min_dist, int &gravity_device_id, double &missile_cost, int &hit_time_step) { 

    cudaSetDevice(0);
    // basic
    double4 *q, *v;
    double4 *dq, *dv;
    double *hmin_dist, *dmin_dist;
    double *hcost, *dcost;
    int *hisHit, *disHit;
    
    q = new double4[n];
    v = new double4[n];
    hmin_dist = new double;
    *hmin_dist = std::numeric_limits<double>::infinity();
    hcost = new double;
    *hcost = 0;
    hisHit = new int;
    *hisHit = -1;

    read_input(filename, n, q, v);

    double4 tmp;
    // planet
    tmp = q[planet];
    q[planet] = q[0];
    q[0] = tmp;
    tmp = v[planet];
    v[planet] = v[0];
    v[0] = tmp;
    planet = 0;

    // asteroid
    tmp = q[asteroid];
    q[asteroid] = q[1];
    q[1] = tmp;
    tmp = v[asteroid];
    v[asteroid] = v[1];
    v[1] = tmp;
    asteroid = 1;

    cudaMalloc(&dq, n * sizeof(double4));
    cudaMalloc(&dv, n * sizeof(double4));
    cudaMalloc(&dmin_dist, sizeof(double));
    cudaMalloc(&dcost, sizeof(double));
    cudaMalloc(&disHit, sizeof(int));

    const int numOfThreads = NumOfThreads;
    const int numOfBlocks = n;
    const int numOfBlocks2 = (n + NumOfThreads - 1) / NumOfThreads;

    double m_orig[n];

    for (int i=0; i<n; i++) {
        m_orig[i] = q[i].w;
        if (v[i].w == Device) {
            q[i].w = 0;
        }
    }

    // cudamemcpy
    cudaMemcpy(dq, q, n * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, n * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(dmin_dist, hmin_dist, sizeof(double), cudaMemcpyHostToDevice);

    min_dist = std::numeric_limits<double>::infinity();

    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<numOfBlocks, numOfThreads>>>(step, n, dq, dv);
            cudaDeviceSynchronize();

            update_position<<<numOfBlocks2, numOfThreads>>>(n, dq, dv, dmin_dist);
            cudaDeviceSynchronize();
        }
    }
    cudaMemcpy(hmin_dist, dmin_dist, sizeof(double), cudaMemcpyDeviceToHost);
    min_dist = *hmin_dist;
    
    for (int i = 0; i < n; i++) {
        q[i].w = m_orig[i];
    }

    for (auto i : todo) {
        // device
        tmp = q[i];
        q[i] = q[2];
        q[2] = tmp;
        tmp = v[i];
        v[i] = v[2];
        v[2] = tmp;

        *hcost = 0;
        *hisHit = -1;

        // cudamemcpy
        cudaMemcpy(dq, q, n * sizeof(double4), cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v, n * sizeof(double4), cudaMemcpyHostToDevice);
        cudaMemcpy(dcost, hcost, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(disHit, hisHit, sizeof(int), cudaMemcpyHostToDevice);

        bool isHit = false;
        bool getCost = false;
        double cost = std::numeric_limits<double>::infinity();            

        for (int step = 0; step <= param::n_steps; step++) {
            run_step3<<<numOfBlocks, numOfThreads>>>(step, n, dq, dv, disHit);
            cudaDeviceSynchronize();

            update_position3<<<numOfBlocks2, numOfThreads>>>(n, step, dq, dv, disHit, dcost);
            cudaDeviceSynchronize();
        }
        
        cudaMemcpy(hisHit, disHit, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hcost, dcost, sizeof(double), cudaMemcpyDeviceToHost);

        if (*hisHit == -1) {
            if (missile_cost == 0 || missile_cost > *hcost) {
                missile_cost = *hcost;
                gravity_device_id = i;
            }
        }
        q[i].w = m_orig[i];
        cudaMemcpy(&dq[i].w, &q[i].w, sizeof(double), cudaMemcpyHostToDevice);
    }

    delete[] q;
    delete[] v;
    cudaFree(dq);
    cudaFree(dv);

    delete hmin_dist;
    cudaFree(dmin_dist);
}

void problem2(char *filename, int n, int planet, int asteroid, std::vector<int> todo, int &hit_time_step, int &gravity_device_id, double &missile_cost) {

    cudaSetDevice(1);
    // basic
    double4 *q, *v;
    double4 *dq, *dv;
    int *hhit_time_step, *dhit_time_step;
    double *hcost, *dcost;
    int *hisHit, *disHit;
    
    q = new double4[n];
    v = new double4[n];
    hhit_time_step = new int;
    *hhit_time_step = -2;
    hcost = new double;
    *hcost = 0;
    hisHit = new int;
    *hisHit = -1;

    read_input(filename, n, q, v);

    double4 tmp;
    // planet
    tmp = q[planet];
    q[planet] = q[0];
    q[0] = tmp;
    tmp = v[planet];
    v[planet] = v[0];
    v[0] = tmp;
    planet = 0;

    // asteroid
    tmp = q[asteroid];
    q[asteroid] = q[1];
    q[1] = tmp;
    tmp = v[asteroid];
    v[asteroid] = v[1];
    v[1] = tmp;
    asteroid = 1;

    cudaMalloc(&dq, n * sizeof(double4));
    cudaMalloc(&dv, n * sizeof(double4));
    cudaMalloc(&dhit_time_step, sizeof(int));
    cudaMalloc(&dcost, sizeof(double));
    cudaMalloc(&disHit, sizeof(int));

    const int numOfThreads = NumOfThreads;
    const int numOfBlocks = n;
    const int numOfBlocks2 = (n + NumOfThreads - 1) / NumOfThreads;

    // cudamemcpy
    cudaMemcpy(dq, q, n * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, n * sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(dhit_time_step, hhit_time_step, sizeof(int), cudaMemcpyHostToDevice);

    // Problem 2
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step2<<<numOfBlocks, numOfThreads>>>(step, n, dq, dv, dhit_time_step);
            cudaDeviceSynchronize();

            update_position2<<<numOfBlocks2, numOfThreads>>>(n, step, dq, dv, dhit_time_step);
            cudaDeviceSynchronize();
        }
    }
    cudaMemcpy(hhit_time_step, dhit_time_step, sizeof(int), cudaMemcpyDeviceToHost);
    hit_time_step = *hhit_time_step;

    // // Problem 3
    double m_orig[n];
    for (int i = 0; i < n; i++) {
        m_orig[i] = q[i].w;
    }

    if (hit_time_step != -2) {
        for (auto i : todo) {
            tmp = q[i];
            q[i] = q[2];
            q[2] = tmp;
            tmp = v[i];
            v[i] = v[2];
            v[2] = tmp;

            *hcost = 0;
            *hisHit = -1;

            // cudamemcpy
            cudaMemcpy(dq, q, n * sizeof(double4), cudaMemcpyHostToDevice);
            cudaMemcpy(dv, v, n * sizeof(double4), cudaMemcpyHostToDevice);
            cudaMemcpy(dcost, hcost, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(disHit, hisHit, sizeof(int), cudaMemcpyHostToDevice);

            bool isHit = false;
            bool getCost = false;
            double cost = std::numeric_limits<double>::infinity();            

            for (int step = 0; step <= param::n_steps; step++) {
                run_step3<<<numOfBlocks, numOfThreads>>>(step, n, dq, dv, disHit);
                cudaDeviceSynchronize();

                update_position3<<<numOfBlocks2, numOfThreads>>>(n, step, dq, dv, disHit, dcost);
                cudaDeviceSynchronize();
            }

            cudaMemcpy(hisHit, disHit, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hcost, dcost, sizeof(double), cudaMemcpyDeviceToHost);

            if (*hisHit == -1) {
                if (missile_cost == 0 || missile_cost > *hcost) {
                    missile_cost = *hcost;
                    gravity_device_id = i;
                }
            }

            q[i].w = m_orig[i];
            cudaMemcpy(&dq[i].w, &q[i].w, sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    delete[] q;
    delete[] v;
    cudaFree(dq);
    cudaFree(dv);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    read_basic(argv[1], n, planet, asteroid);

    double4 *q, *v;
    q = new double4[n];
    v = new double4[n];

    read_input(argv[1], n, q, v);

    std::vector<int> cal_t1, cal_t2;
    int turn = -1;
    for (int i = 0; i < n; i++) {
        if (v[i].w != Device) continue;

        if (turn == 1) cal_t1.push_back(i);
        else cal_t2.push_back(i);
        turn *= (-1);
    }

    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    int gravity_device_id = -1;
    double missile_cost = 0;

    int gravity_device_id1, gravity_device_id2;
    gravity_device_id1 = gravity_device_id2 = -1;
    double missile_cost1, missile_cost2;
    missile_cost1 = missile_cost2 = 0;

    std::thread t1(problem1, argv[1], n, planet, asteroid, cal_t1, std::ref(min_dist), std::ref(gravity_device_id1), std::ref(missile_cost1), std::ref(hit_time_step));
    std::thread t2(problem2, argv[1], n, planet, asteroid, cal_t2, std::ref(hit_time_step), std::ref(gravity_device_id2), std::ref(missile_cost2));

    t1.join();
    t2.join();

    if (hit_time_step != -2) {
        if (missile_cost1 < missile_cost2) {
            missile_cost = missile_cost1;
            gravity_device_id = gravity_device_id1;
        }
        else {
            missile_cost = missile_cost2;
            gravity_device_id = gravity_device_id2;
        }
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
