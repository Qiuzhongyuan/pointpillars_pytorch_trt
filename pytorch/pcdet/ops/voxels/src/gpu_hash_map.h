#ifndef GPU_HASH_MAP_
#define GPU_HASH_MAP_
#include <vector>
#include <cuda_fp16.h>
#define NAMESPACE VoxelGeneratorSpace
#define SHARED_MEM 49152
namespace NAMESPACE
{
#if defined(DEBUG) || defined(_DEBUG)
#define checkCudaErrors(x) do { \
  cudaError_t result = x;\
  if (result != cudaSuccess) { \
    fprintf(stderr, "CUDA Runtime Error at %s\nerror type: %s\nerror path: %s:%d\n", #x, cudaGetErrorString(result), __FILE__, __LINE__); \
    exit(-1);\
  } \
} while(0)
#else
    #define checkCudaErrors(x) cudaError_t result = x
#endif

static inline int DivUp(int a, int b) { return (a + b - 1) / b; }
struct __align__(4) HashEntry
{
    int nextId;
    int4 intCoor;
};


void InitializeHashMap(const float* points, const int* numValid, int* validOutputVoxels, int* map, HashEntry* list, int* map_addr,
        int batchSize, int maxInputPoints, int maxOutputVoxels, int inCols,
        std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size, const int value_map_z);

void InitializeHashMapFp16(const __half* points, const int* numValid, int* validOutputVoxels, int* map, HashEntry* list, int* map_addr,
                int batchSize, int maxInputPoints, int maxOutputVoxels, int inCols,
                std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size, const int value_map_z);

} // namespace

#endif