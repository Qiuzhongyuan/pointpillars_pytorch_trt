#ifndef GPU_HASH_MAP_v1
#define GPU_HASH_MAP_v1
#include <vector>
#include <cuda_fp16.h>
#define NAMESPACE VoxelGeneratorV1Space
#define SHARED_MEM 49152 //65536  49152
namespace NAMESPACE
{
#define MAX_THREAD_PER_BLOCK 1024
inline
cudaError_t checkCudaErrors(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
static inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct __align__(4) HashEntry
{
    int nextId;
    int3 intCoor;
};


void InitializeHashMap(const float* points, const int* numValid, int* validOutputVoxels, int* map, HashEntry* list, int* map_addr,
        int batchSize, int maxInputPoints, int maxOutputVoxels, int inCols,
        std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size, const int value_map_z);

void InitializeHashMapFp16(const __half* points, const int* numValid, int* validOutputVoxels, int* map, HashEntry* list, int* map_addr,
                int batchSize, int maxInputPoints, int maxOutputVoxels, int inCols,
                std::vector<float> point_cloud_range, std::vector<float> voxel_size, std::vector<int> grid_size, const int value_map_z);


} // namespace

#endif