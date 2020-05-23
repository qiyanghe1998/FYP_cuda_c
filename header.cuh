#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <cassert>
#include <thread>
#include <utility>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <functional>
#include <cub/cub.cuh>
#define CHECK(call) { const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)) exit(1);}}

using std::cout;
using std::cerr;
using std::endl;
using std::stringstream;
using std::fstream;
using std::ifstream;
using std::ofstream;
using std::setprecision;

using std::string;
using std::vector;
using std::set;
using std::list;
using std::to_string;
using std::nth_element;

using std::size_t;