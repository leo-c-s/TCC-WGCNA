
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cusolverDn.h>

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>

using namespace std::literals::chrono_literals;

#include "Matrix.hpp"
#include "Tree.hpp"

#define MIN(x, y) ((x) < (y) ? (x) : (y))

__global__ void averagesKernel(float* const result, float* const X, int m, int n)
{
    int const rowIdx = blockIdx.x * blockIdx.x + threadIdx.x;

    if (rowIdx >= n) return;

    float avg = 0.0;
    for (int i = 0; i < m; i++)
    {
        avg += X[rowIdx * m + i] / m;
    }

    result[rowIdx] = avg;
}

__global__ void standardDeviationKernel(float* const result, float* const X, float* const averages, int m, int n)
{
    int const rowIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx >= n) return;

    float var = 0.0;
    for (int i = 0; i < m; i++)
    {
        float x = X[rowIdx * m + i] - averages[rowIdx];
        var += x * x / m;
    }

    result[rowIdx] = sqrt(var);
}

__global__ void correlationKernel(float* const C, float* const averages, float* const std, float* const X, int m, int n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int const j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    // if variance is zero, assume no correlation
    if (abs(std[i]) < 1e-15 || abs(std[j]) < 1e-15)
    {
        C[i * n + j] = 0.0f;
        return;
    }

    float corr = 0.0;

    for (int k = 0; k < m; k++)
    {
        corr += (X[i * m + k] - averages[i]) * (X[j * m + k] - averages[j]) / (m - 1);
    }

    corr /= std[i] * std[j];

    C[i * n + j] = corr;
}

__global__ void adjacencyKernel(float* const Adj, float* const Similarity, int n, int power, bool useSignedSimilarity)
{
    int const i = blockIdx.x;
    int const j = blockIdx.y;

    float adj = Similarity[i * n + j];

    if (useSignedSimilarity)
    {
        adj = 0.5f + adj / 2.0f;
    }
    
    Adj[i * n + j] = pow(abs(adj), power);
}

__global__ void topologicalOverlapKernel(float* const TOM, float* const Adj, int n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int const j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    if (i == j)
    {
        TOM[i * n + j] = 1.0f;
        return;
    }

    float l = 0.0f;
    float ki = 0.0f;
    float kj = 0.0f;

    for (int u = 0; u < n; u++)
    {
        if (u == i || u == j) continue;

        l += Adj[i * n + u] * Adj[j * n + u];
        ki += Adj[i * n + u];
        kj += Adj[j * n + u];
    }

    TOM[i * n + j] = (l - Adj[i * n + j]) / (MIN(ki, kj) + 1 - Adj[i * n + j]);
}

__global__ void dissTom(float* dissTOM, float* const TOM, int n)
{
    int const i = blockIdx.x;
    int const j = blockIdx.y;

    dissTOM[i * n + j] = 1 - TOM[i * n + j];
}

__global__ void minDistanceKernel(float* minDistances, int* minIdxs, float* Dist, int numClusters)
{
    int const clusterIdx = blockIdx.x;

    float minDist = 1.0f;
    int minDistIdx = 0;

    for (int i = 0; i < numClusters; i++)
    {
        if (i == clusterIdx) continue;

        float const dist = Dist[clusterIdx * numClusters + i];
        if (dist < minDist)
        {
            minDist = dist;
            minDistIdx = i;
        }
    }

    minDistances[clusterIdx] = minDist;
    minIdxs[clusterIdx] = minDistIdx;
}

__global__ void mergeClustersKernel(float* newDist, float* Dist, int n, int first, int second, int* clusterSizes)
{
    int const clusterIdx = blockIdx.x;
    int const targetIdx = blockIdx.y;

    int i = clusterIdx, j = targetIdx;

    if (clusterIdx >= second) {
        i += 1;
    }

    if (targetIdx >= second)
    {
        j += 1;
    }

    if (clusterIdx == first)
    {
        float const sumDistsFirst = clusterSizes[first] * Dist[first * n + j];
        float sumDistsSecond = clusterSizes[second] * Dist[second * n + j];
        int mergedClusterSize = clusterSizes[first] + clusterSizes[second];

        newDist[clusterIdx * (n - 1) + targetIdx] = (sumDistsFirst + sumDistsSecond) / mergedClusterSize;
    }
    else if (targetIdx == first)
    {
        float sumDistsFirst = clusterSizes[first] * Dist[first * n + i];
        float sumDistsSecond = clusterSizes[second] * Dist[second * n + i];
        int mergedClusterSize = clusterSizes[first] + clusterSizes[second];

        newDist[clusterIdx * (n - 1) + targetIdx] = (sumDistsFirst + sumDistsSecond) / mergedClusterSize;
    }
    else
    {
        newDist[clusterIdx * (n - 1) + targetIdx] = Dist[i * n + j];
    }
}

template<typename T>
static inline T* cuda_new(std::size_t count) {
    T* ptr;

#ifndef NDEBUG
    cudaError_t status = cudaMalloc(&ptr, count * sizeof(T));
    if (status != cudaSuccess)
    {
        throw "CUDA failed to allocate";
    }
#else
    cudaMalloc(&ptr, count * sizeof(T));
#endif

    return ptr;
}

template<typename T>
static inline void cuda_memcpy(T* dst, T const* src, std::size_t count, cudaMemcpyKind flag)
{
#ifndef NDEBUG
    cudaError_t status = cudaMemcpy(dst, src, count * sizeof(T), flag);
    if (status != cudaSuccess)
    {
        throw "CUDA failed to copy";
    }
#else
    cudaMemcpy(dst, src, count * sizeof(T), flag);
#endif
}

struct Bucket
{
    float midpoint;
    float probability;
};

struct Distribution
{
    std::vector<Bucket> buckets;
    int numSamples;
};

static Distribution distribution(Matrix Adj, int numBuckets = 10)
{
    int const n = Adj.Size().first;

    std::vector<float> degrees;
    degrees.reserve(n);

    for (int i = 0; i < n; i++)
    {
        float degree = -Adj(i,i);

        for (int j = 0; j < n; j++)
        {
            degree += Adj(i, j);
        }

        degrees.push_back(degree);
    }

    std::sort(degrees.begin(), degrees.end());
  
    float const base = degrees.front();
    float const step = (degrees.back() - base) / numBuckets;

    std::vector<Bucket> dist;
    dist.reserve(numBuckets);
    
    auto it = degrees.begin();
    for (int i = 0; i < numBuckets - 1; i++)
    {
        float const endpoint = base + (i + 1) * step;

        auto const end = std::find_if(it, degrees.end(), [endpoint](auto x) { return x >= endpoint; });

        float const relativeFrequency = static_cast<float>(std::distance(it, end)) / n;

        float midpoint = endpoint - step / 2.f;

        dist.push_back(Bucket { midpoint, relativeFrequency });

        it = end;
    }

    { // Last bucket
        float midpoint = base + numBuckets * step - step / 2.f;
        float relativeFrequency = static_cast<float>(std::distance(it, degrees.end())) / n;

        dist.push_back(Bucket{ midpoint, relativeFrequency });
    }

    return Distribution{ dist, n };
}

// Computes R^2 as the square of the correlation between actual and predicted values
static float checkScaleFreeFit(Distribution const& dist)
{
    // E[log(p(k))]; E[log(k)]
    std::pair<float,float> const means = std::accumulate(
        dist.buckets.begin(),
        dist.buckets.end(),
        std::pair<double,double>{ 0.0, 0.0 },
        [&](auto acc, auto b)
        {
            return std::pair<double,double> {
                acc.first + std::log10(b.probability + 1e-9) / dist.buckets.size(),
                acc.second + std::log10(b.midpoint) / dist.buckets.size()
            };  
        });
    
    // Var[log(p(k))]; Var[log(k)]; Cov[log(p(k)), log(k)]
    std::tuple<float, float, float> const vars = std::accumulate(
        dist.buckets.begin(),
        dist.buckets.end(),
        std::tuple<double, double, double>{ 0.0, 0.0, 0.0 },
        [&](auto acc, auto b)
        {
            float const x = std::log10(b.probability + 1e-9) - means.first;
            float const y = std::log10(b.midpoint) - means.second;

            int const size = dist.buckets.size() - 1;

            return std::tuple<double, double, double> {
                std::get<0>(acc) + x * x / size,
                std::get<1>(acc) + y * y / size,
                std::get<2>(acc) + x * y / size
            };
        });

    float const correlation = std::get<2>(vars) / std::sqrt(std::get<0>(vars) * std::get<1>(vars));

    float rSquared = correlation * correlation;
    if (correlation > 0.0f)
    {
        rSquared *= -1.0f;
    }

    return rSquared;
}

static std::tuple<Matrix, int, float> pickSoftThreshold(float* Similarity, int n)
{
    cudaError_t status;

    Matrix Adj(n, n);
    Matrix bestAdj(n, n);

    float* devAdj = cuda_new<float>(n * n);

    float* devSimilarity = cuda_new<float>(n * n);

    cuda_memcpy(devSimilarity, Similarity, n * n, cudaMemcpyHostToDevice);
    
    dim3 const dim(n, n);

    adjacencyKernel<<<dim, 1>>>(devAdj, devSimilarity, n, 1, true);
    status = cudaDeviceSynchronize();
#ifndef NDEBUG
    if (status != cudaSuccess)
    {
        throw "adjacencyKernel failed";
    }
#endif

    cuda_memcpy(Adj.Data(), devAdj, n * n, cudaMemcpyDeviceToHost);

    auto dist = distribution(Adj);

    float bestRSquared = checkScaleFreeFit(dist);
    int bestPower = 1;

    for (int power = 6; power <= 6; power += 2)
    {
        adjacencyKernel<<<dim, 1>>>(devAdj, devSimilarity, n, power, true);
        status = cudaDeviceSynchronize();
#ifdef DEBUG
        if (status != cudaSuccess)
        {
            throw "adjacencyKernel failed";
        }
#endif
        cuda_memcpy(Adj.Data(), devAdj, n * n, cudaMemcpyDeviceToHost);
        dist = distribution(Adj);

        float rSquared = checkScaleFreeFit(dist);

        if (rSquared > bestRSquared)
        {
            bestRSquared = rSquared;
            bestPower = power;
            
            bestAdj = Adj;
        }
    }

    cudaFree(devAdj);
    cudaFree(devSimilarity);

    return { bestAdj, bestPower, bestRSquared };
}

template<typename T>
int argmin(std::vector<T> values)
{
    int minIdx = 0;
    T min = values[0];

    for (int i = 0; i < values.size(); i++)
    {
        if (values[i] < min)
        {
            min = values[i];
            minIdx = i;
        }
    }

    return minIdx;
}

Tree* cluster(Matrix dist) {
    int n = dist.Size().first;

    std::vector<Tree*> clusterTrees(n);
    for (int i = 0; i < n; i++)
    {
        clusterTrees[i] = new Tree(i);
    }
    std::vector<int> clusterSizes(n, 1);

    int const bufferSize = n * n;

    cudaError_t status;
    
    float* devDist = cuda_new<float>(bufferSize);
    
    cuda_memcpy(devDist, dist.Data(), bufferSize, cudaMemcpyHostToDevice);

    float* devResult = cuda_new<float>(bufferSize);

    cudaMemset(devResult, 0, bufferSize * sizeof(float));

    int* devClusterSizes = cuda_new<int>(n);

    cuda_memcpy(devClusterSizes, clusterSizes.data(), n, cudaMemcpyHostToDevice);

    float* devMinDistances = cuda_new<float>(n);

    int* devMinIdxs = cuda_new<int>(n);

    while (clusterTrees.size() > 1)
    {
        minDistanceKernel<<<n, 1>>>(devMinDistances, devMinIdxs, devDist, n);
        status = cudaDeviceSynchronize();
#ifndef NDEBUG
        if (status != cudaSuccess) {
            throw;
        }
#endif

        std::vector<float> minDistances(n);
        std::vector<int> minDistanceIdxs(n);

        cuda_memcpy(minDistances.data(), devMinDistances, n, cudaMemcpyDeviceToHost);

        cuda_memcpy(minDistanceIdxs.data(), devMinIdxs, n, cudaMemcpyDeviceToHost);

        int minIdx = argmin(minDistances);
        int mergeTarget = minDistanceIdxs[minIdx];

        int firstCluster = std::min(minIdx, mergeTarget);
        int secondCluster = std::max(minIdx, mergeTarget);

        cuda_memcpy(devClusterSizes, clusterSizes.data(), n, cudaMemcpyHostToDevice);

        dim3 dim(n, n);
        mergeClustersKernel<<<dim, 1>>>(
            devResult,
            devDist,
            n,
            firstCluster,
            secondCluster,
            devClusterSizes
        );
        
        status = cudaDeviceSynchronize();
#ifndef NDEBUG
        if (status != cudaSuccess) {
            throw;
        }
#endif

        clusterTrees[firstCluster] = new Tree(minDistances[minIdx], clusterTrees[firstCluster], clusterTrees[secondCluster]);
        clusterTrees.erase(clusterTrees.begin() + secondCluster);
        
        clusterSizes[firstCluster] += clusterSizes[secondCluster];
        clusterSizes.erase(clusterSizes.begin() + secondCluster);

        std::swap(devDist, devResult);
        n--;
    }

    return clusterTrees.front();
}

std::map<int, int> wgcna(Matrix const& testData)
{
    auto const size = testData.Size();
    int const numSamples = size.second;
    int const numVars = size.first;

    float const* X = testData.Data();

    cudaError_t status;

    float* devX = cuda_new<float>(numVars * numSamples);

    cuda_memcpy(devX, X, numVars * numSamples, cudaMemcpyHostToDevice);

    float* devAverages = cuda_new<float>(numVars);
    float* devStds = cuda_new<float>(numVars);
    float* devC = cuda_new<float>(numVars * numVars);

    {
        int numThreadsPerBlock = std::min(numVars, 1024);
        int numBlocks = numVars / numThreadsPerBlock + static_cast<int>(numVars % numThreadsPerBlock > 0);

        averagesKernel<<<numBlocks, numThreadsPerBlock>>>(devAverages, devX, numSamples, numVars);
    }

    status = cudaDeviceSynchronize();
#ifndef NDEBUG
    if (status != cudaSuccess)
    {
        std::cerr << "Error code: " << status << std::endl;
        throw "averagesKernel failed";
    }
#endif

    {
        int numThreadsPerBlock = std::min(numVars, 1024);
        int numBlocks = numVars / numThreadsPerBlock + static_cast<int>(numVars % numThreadsPerBlock > 0);

        standardDeviationKernel<<<numBlocks, numThreadsPerBlock>>>(devStds, devX, devAverages, numSamples, numVars);
    }

    status = cudaDeviceSynchronize();

#ifndef NDEBUG
    if (status != cudaSuccess)
    {
        throw "standardDeviationKernel failed";
    }
#endif

    {
        int numThreads1D = std::min(numVars, 32);
        int numBlocks1D = numVars / numThreads1D + static_cast<int>(numVars / numThreads1D > 0);

        dim3 dimBlocks(numBlocks1D, numBlocks1D);
        dim3 dimThreads(numThreads1D, numThreads1D);
        correlationKernel<<<dimBlocks, dimThreads>>>(devC, devAverages, devStds, devX, numSamples, numVars);
    }

    status = cudaDeviceSynchronize();

#ifndef NDEBUG
    if (status != cudaSuccess)
    {
        throw "correlationKernel failed";
    }
#endif

    cudaFree(devAverages);
    cudaFree(devStds);
    cudaFree(devX);

    float* C = cuda_new<float>(numVars * numVars);

    cuda_memcpy(C, devC, numVars * numVars, cudaMemcpyDeviceToHost);

    Matrix Adjacency;
    int thresholdPower;
    float rSquared;
    {
        auto res = pickSoftThreshold(C, numVars);
        Adjacency = std::get<0>(res);
        thresholdPower = std::get<1>(res);
        rSquared = std::get<2>(res);
    }

    float* devAdj = cuda_new<float>(numVars * numVars);

    cuda_memcpy(devAdj, Adjacency.Data(), numVars * numVars, cudaMemcpyHostToDevice);

    float* devTOM = cuda_new<float>(numVars * numVars);

    {
        int numThreads1D = std::min(numVars, 32);
        int numBlocks1D = numVars / numThreads1D + static_cast<int>(numVars / numThreads1D > 0);

        dim3 dimBlocks(numBlocks1D, numBlocks1D);
        dim3 dimThreads(numThreads1D, numThreads1D);
        topologicalOverlapKernel<<<dimBlocks, dimThreads>>>(devTOM, devAdj, numVars);
    }
    status = cudaDeviceSynchronize();

#ifndef NDEBUG
    if (status != cudaSuccess)
    {
        throw "topologicalOverlapKernel failed";
    }
#endif

    float* devDissTOM = devTOM;
    {
        int numThreads1D = std::min(numVars, 32);
        int numBlocks1D = numVars / numThreads1D + static_cast<int>(numVars / numThreads1D > 0);

        dim3 dimBlocks(numBlocks1D, numBlocks1D);
        dim3 dimThreads(numThreads1D, numThreads1D);
        dissTom<<<dimBlocks,dimThreads>>>(devDissTOM, devTOM, numVars);
    }

    status = cudaDeviceSynchronize();
#ifndef NDEBUG
    if (status != cudaSuccess)
    {
        throw "Failed to compute dissTOM";
    }
#endif

    Matrix dissTOM(numVars, numVars);
    cuda_memcpy(dissTOM.Data(), devDissTOM, numVars* numVars, cudaMemcpyDeviceToHost);

    Tree* dendro = cluster(dissTOM);
    auto clusterMap = dendro->DynamicCut(0.5 * dendro->Dist());

    return clusterMap;
}

std::vector<float> getEigengene(Matrix A)
{
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cusolverDnSetStream(handle, stream);

    cusolverDnParams_t params = nullptr;
    cusolverDnCreateParams(&params);

    std::size_t workspaceInBytesOnDevice = 0;
    std::size_t workspaceInBytesOnHost = 0;

    std::size_t const m = A.Size().first;
    std::size_t const n = A.Size().second;

    float* devA = cuda_new<float>(m * n);
    cuda_memcpy(devA, A.Data(), m * n, cudaMemcpyHostToDevice);

    float* devU = cuda_new<float>(m * m);
    float* devVt = cuda_new<float>(n * n);
    float* devSingularValues = cuda_new<float>(m);

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    status = cusolverDnXgesvd_bufferSize(
        handle,
        params,
        'A',
        'A',
        m,
        n,
        CUDA_R_32F,
        devA,
        n,
        CUDA_R_32F,
        devSingularValues,
        CUDA_R_32F,
        devU,
        m,
        CUDA_R_32F,
        devVt,
        n,
        CUDA_R_32F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    );

    char* workspace = new char[workspaceInBytesOnHost];

    void* devWorkspace = cuda_new<char>(workspaceInBytesOnDevice);

    int* devInfo = cuda_new<int>(1);

    status = cusolverDnXgesvd(
        handle,
        params,
        'A',
        'A',
        m,
        n,
        CUDA_R_32F,
        devA,
        n,
        CUDA_R_32F,
        devSingularValues,
        CUDA_R_32F,
        devU,
        m,
        CUDA_R_32F,
        devVt,
        n,
        CUDA_R_32F,
        devWorkspace,
        workspaceInBytesOnDevice,
        workspace,
        workspaceInBytesOnHost,
        devInfo
    );

    cudaDeviceSynchronize();

    Matrix U(m, m);

    // copy only the first row of devVt, which contains the first eigenvector of A
    cuda_memcpy(U.Data(), devU, m * m, cudaMemcpyDeviceToHost);

    std::vector<float> eigenvalues(m);
    cuda_memcpy(eigenvalues.data(), devSingularValues, m, cudaMemcpyDeviceToHost);
   
    cudaDeviceSynchronize();

    std::vector<float> eigengene(m);
    for (int i = 0; i < m; i++)
    {
        eigengene[i] = U(i, 0);
    }

    return eigengene;
}

std::vector<float> getClusterEigengene(std::map<int, int> clusterNodes, int targetCluster, Matrix exprData)
{
    std::vector<int> nodes;

    for (auto kp : clusterNodes)
    {
        int node = kp.first;
        int cluster = kp.second;

        if (cluster == targetCluster)
        {
            nodes.push_back(node);
        }
    }

    std::sort(nodes.begin(), nodes.end());
    
    int const m = nodes.size();
    int const n = exprData.Size().second;

    Matrix clusterExpr(n, m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        {
            clusterExpr(j, i) = exprData(nodes[i], j);
        }
    }

    return getEigengene(clusterExpr);
}

int main(int argc, char *argv[])
{
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    std::string inputPath{ argv[1] };
    Matrix inputData = read_csv(inputPath);
    auto clusterMap = wgcna(inputData);

    int numClusters = 1;

    std::ofstream ofile("results.csv");
    ofile << "node,cluster\n";

    for (auto kp : clusterMap)
    {
        int node = kp.first;
        int cluster = kp.second;

        if (cluster > numClusters) numClusters = cluster;

        ofile << node << "," << cluster << '\n';
    }

    std::ofstream eigengenesFile("eigengenes.csv");

    for (int cluster = 1; cluster <= numClusters; cluster++)
    {
        auto eigengene = getClusterEigengene(clusterMap, cluster, inputData);

        eigengenesFile << eigengene[0];
    
        for (int i = 1; i < eigengene.size(); i++)
        {
            eigengenesFile << ',' << eigengene[i];
        }
        eigengenesFile << '\n';
    }

    return 0;
}
