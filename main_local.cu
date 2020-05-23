#include "header.cuh"


typedef float pq_float;
typedef uint8_t pq_int;
// configuration 
int topk;
int codebook;
int Ks;
int group;
int batch_vecs;
int batch_query;
int threads_per_block;
// size of everything
int size_codeword_nq;
int size_codeword_pq;
int size_codebook;
int size_query;
int size_threshold;
int size_device_query;
int size_vecs;
int size_norm_filter;
int size_q_map;
int size_lookup_table;
int size_ret_result;
string dataset;

int num_vecs, num_dimen, num_query, num_q, top_norm;
pq_float *vecs;
pq_float *query;
int *ret_result;
int query_idx;

int cal_2Dcoordinate(int x, int y, int leny) {
    return x * leny + y;
}

int cal_3Dcoordinate(int x, int y, int z, int leny, int lenz) {
    return x * leny * lenz + y * lenz + z;
}

__global__ void calLookupOnGPU(pq_float *query, pq_float *codeword_nq, pq_float *codeword_pq, pq_int *q_map, pq_float *lookup_table, 
    int num_dimen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_type = q_map[2 * blockIdx.x];
    int idx_q = q_map[2 * blockIdx.x + 1];
    if (q_type == 0) {
        lookup_table[idx] = codeword_nq[idx_q * blockDim.x + threadIdx.x];
    } else if (q_type == 1) {
        pq_float temp_sum = 0;
        for (int i = 0; i < num_dimen; ++i) {
            temp_sum += codeword_pq[idx_q * blockDim.x * num_dimen + threadIdx.x * num_dimen + i] * query[i];
        }
        lookup_table[idx] = temp_sum;
    }
}

void calLookupOnCPU(pq_float *query, pq_float *codeword_nq, pq_float *codeword_pq, pq_int *q_map, pq_float *lookup_table, 
    int num_dimen, int num_q, int Ks) {
    for (int i = 0; i < num_q; ++i) {
        int q_type = q_map[2 * i];
        int idx_q = q_map[2 * i + 1];
        for (int j = 0; j < Ks; ++j) {
            if (q_type == 0) {
                lookup_table[cal_2Dcoordinate(i, j, Ks)] = codeword_nq[cal_2Dcoordinate(idx_q, j, Ks)];
            } else if (q_type == 1) {
                pq_float temp_sum = 0;
                for (int k = 0; k < num_dimen; ++k) temp_sum += codeword_pq[cal_3Dcoordinate(idx_q, j, k, Ks, num_dimen)] * query[k];
                lookup_table[cal_2Dcoordinate(i, j, Ks)] = temp_sum;
            }
        }
    }
}

void checkLookup(pq_float *h_lookup_table, pq_float *gpuref_lookup_table, int length) {
    double eps = 1e-7;
    bool match = 1;
    for (int i = 0; i < length; ++i) {
        if (abs(h_lookup_table[i] - gpuref_lookup_table[i]) > eps) {
            match = 0;
            printf("Lookup tables do not match!\n");
            printf("host %f gpu %f at current %d\n", h_lookup_table[i], gpuref_lookup_table[i], i);
        }
    }
    if (match) printf("Lookup tables match.\n\n");
}

__global__ void calApproxVecs(int *mid_result, pq_int *codebook, pq_float *lookup_table, pq_int *q_map, int num_q, int Ks, int num_vecs,
    int start_vecs, int end_vecs, pq_float threshold) {
    int average_assign = num_q * Ks / blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ pq_float local_lookup_table [];
    for (int i = 0; i < average_assign; ++i) local_lookup_table[threadIdx.x * average_assign + i] = lookup_table[threadIdx.x * average_assign + i];

    __syncthreads();
    // init
    if (start_vecs + idx < end_vecs) {
        int q_type = 0;
        pq_float result = 0, coefficient = 1;
        for (int i = 0; i < num_q; ++i) {
            q_type = q_map[2 * i];
            if (q_type == 0) {
                coefficient = local_lookup_table[i * Ks + codebook[i * num_vecs + (start_vecs + idx)]];
            } else if (q_type == 1) {
                result += coefficient * local_lookup_table[i * Ks + codebook[i * num_vecs + (start_vecs + idx)]];
            }
        }
        if (result >= threshold) mid_result[idx] = 1;
        else mid_result[idx] = 0;
        // mid_result[idx] = result;
    }
}

void calMidResultOnCPU(int *mid_result, pq_int *codebook, pq_float *lookup_table, pq_int *q_map, int num_q, int Ks, int num_vecs, 
    int start_pos, int end_pos, pq_float threshold) {
    for (int i = 0; i < end_pos - start_pos; ++i) {
        pq_float result = 0, mid_coefficient = 1;
        for (int j = 0; j < num_q; ++j) {
            int q_type = q_map[2 * j], idx = j * Ks + codebook[j * num_vecs + (start_pos + i)];
            if (q_type == 0) {
                mid_coefficient = lookup_table[idx];
            } else if (q_type == 1) {
                result += mid_coefficient * lookup_table[idx];
            }
        }
        mid_result[i] = result >= threshold;   
    }
}

void checkMidResult(int *h_mid_result, int *gpuref_mid_result, int length) { 
    bool match = 1;
    // pq_float eps = 1e-7;
    for (int i = 0; i < length; ++i) {
        if (h_mid_result[i] != gpuref_mid_result[i]) {
        // if (abs(h_mid_result[i] - gpuref_mid_result[i]) > eps) {
            match = 0;
            printf("Mid results do not match!\n");
            printf("host %d gpu %d at current %d\n", h_mid_result[i], gpuref_mid_result[i], i);
            // printf("host %f gpu %f at current %d\n", h_mid_result[i], gpuref_mid_result[i], i);
            break;
        }
    }
    if (match) printf("Mid results match.\n\n");
}

__global__ void assignResult(int *prefixsum_result, int *d_ret_result, int start_pos, int end_pos, int current_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (start_pos + idx < end_pos) {
        if (!idx) {
            if (prefixsum_result[idx] == 1) {
                d_ret_result[current_length] = start_pos + idx;
            }
        } else {
            if (prefixsum_result[idx] - prefixsum_result[idx - 1]) {
                d_ret_result[current_length + prefixsum_result[idx] - 1] = start_pos + idx;
            }
        }
    }
}

__global__ void initResult(int *d_ret_result, int size_ret_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_ret_result) d_ret_result[idx] = 0;
}

float calNorm(pq_float *arr, int vecs_idx, int query_idx, int num_dimen) {
    pq_float temp_sum = 0;
    int offset_vecs = vecs_idx * num_dimen, offset_query = query_idx * num_dimen;
    for (int i = 0; i < num_dimen; ++i) {
        temp_sum += arr[offset_vecs + i] * query[offset_query + i];
    }
    return temp_sum;
}

bool cmpIPIndex(int lhs, int rhs) {
    return calNorm(vecs, lhs, query_idx, num_dimen) > calNorm(vecs, rhs, query_idx, num_dimen);
}

int sameIndex(int *vecs1, int *vecs2, int number) {
    bool *flag = new bool[number];
    for (int i = 0; i < number; ++i) {
        flag[i] = false;
    }
    int ret = 0;
    for (int i = 0; i < number; ++i) {
        for (int j = 0; j < number; ++j) {
            if (!flag[j] && vecs1[i] == vecs2[j]) {
                flag[j] = true;
                ret++;
            }
        }
    }
    delete [] flag;
    return ret;
}

int main(int argc, char *argv[]) {
    // input configre 
    dataset = string(argv[1]);
    topk = atoi(argv[2]);
    codebook = atoi(argv[3]);
    Ks = atoi(argv[4]);
    group = atoi(argv[5]);
    batch_vecs = atoi(argv[6]);
    batch_query = atoi(argv[7]);
    threads_per_block = atoi(argv[8]);

    // init input data
    printf("# Begin reading data\n");
    string file_path = "../data/" + dataset + "/" + dataset + "_cuda_c.txt";
    freopen(file_path.c_str(), "r", stdin);

    vector<int>q_type;
    pq_int q_map[100];
    scanf("%d%d%d%d", &num_vecs, &num_dimen, &num_query, &top_norm);
    num_q = codebook * group;

    char temp_str[50];
    scanf("%s", temp_str);
    int num_nq = 0, num_pq = 0;
    for (int i = 0; i < num_q; ++i) {
        if (temp_str[i] == 'N') {
            q_type.push_back(0);
            q_map[2 * i] = 0;
            q_map[2 * i + 1] = num_nq;
            num_nq++;
        }
        else if (temp_str[i] == 'P') {
            q_type.push_back(1);
            q_map[2 * i] = 1;
            q_map[2 * i + 1] = num_pq;
            num_pq++;
        }
    }
    // may be larger than the limit of int 
    printf("# Begin allocating memory\n");
    size_codeword_nq = num_nq * Ks;
    size_codeword_pq = num_pq * Ks * num_dimen;
    size_codebook = num_q * num_vecs;
    size_query = num_query * num_dimen;
    size_threshold = num_query;
    size_device_query = batch_query * num_dimen;
    size_vecs = num_vecs * num_dimen;
    size_norm_filter = num_vecs / batch_vecs;
    size_q_map = num_q * 2;
    size_lookup_table = num_q * Ks;
    size_ret_result = num_vecs;

    pq_float *codeword_nq = new pq_float[size_codeword_nq];
    pq_float *codeword_pq = new pq_float[size_codeword_pq];
    pq_int *codebook = new pq_int[size_codebook];
    pq_float *query_norm = new pq_float[num_query];
    pq_float *threshold = new pq_float[size_threshold];
    vecs = new pq_float[size_vecs];
    ret_result = new int[size_ret_result];
    query = new pq_float[size_query];
    pq_float *norm_filter = new pq_float[size_norm_filter + 2];
    pq_float total_time;
    int **candidate_init = new int*[num_query];
    for (int i = 0; i < num_query; ++i) candidate_init[i] = new int[topk];
    int **answer = new int*[num_query];
    for (int i = 0; i < num_query; ++i) answer[i] = new int[topk];

    // load input data
    printf("# Begin reading codeword+codebook\n");
    int temp_nq = 0, temp_pq = 0;
    for (int i = 0; i < num_q; ++i) {
        if (!q_type[i]) {
            for (int j = 0; j < Ks; ++j) scanf("%f", &codeword_nq[cal_2Dcoordinate(temp_nq, j, Ks)]);
            for (int j = 0; j < num_vecs; ++j) scanf("%d", &codebook[cal_2Dcoordinate(temp_nq + temp_pq, j, num_vecs)]);
            temp_nq++;
        } else if (q_type[i] == 1) {
            for (int j = 0; j < Ks; ++j) {
                for (int k = 0; k < num_dimen; ++k) scanf("%f", &codeword_pq[cal_3Dcoordinate(temp_pq, j, k, Ks, num_dimen)]);
            }
            for (int j = 0; j < num_vecs; ++j) scanf("%d", &codebook[cal_2Dcoordinate(temp_nq + temp_pq, j, num_vecs)]); 
            temp_pq++;
        }
    }

    pq_float temp_sum = 0, temp_value = 0;
    printf("# Begin reading query\n");
    for (int i = 0; i < num_query; ++i) {
        temp_sum = 0;
        for (int j = 0; j < num_dimen; ++j) {
            scanf("%f", &temp_value);
            temp_sum += temp_value * temp_value;
            query[cal_2Dcoordinate(i, j, num_dimen)] = temp_value;
        }
        query_norm[i] = sqrt(temp_sum);
    }

    int sum_init_correct = 0;

    for (int i = 0; i < num_query; ++i)
        scanf("%f", &threshold[i]);
    for (int i = 0; i < num_query; ++i) {
        for (int j = 0; j < topk; ++j) {
            scanf("%d", &candidate_init[i][j]);
        }
        for (int j = 0; j < topk; ++j) {
            scanf("%d", &answer[i][j]);
        }
        sum_init_correct += sameIndex(candidate_init[i], answer[i], topk);
    }

    scanf("%f", &total_time);

    int index = 0;
    bool flag = false;
    printf("# Begin reading items\n");
    for (int i = 0; i < num_vecs; ++i) {
        if (i % batch_vecs == top_norm) {
            flag = true;
            temp_sum = 0;
        }
        for (int j = 0; j < num_dimen; ++j) {
            scanf("%f", &temp_value);
            if (flag) temp_sum += temp_value * temp_value;
            vecs[cal_2Dcoordinate(i, j, num_dimen)] = temp_value;
        }
        if (flag) {
            flag = false;
            norm_filter[index] = sqrt(temp_sum);
            index++;
        }
    }

    norm_filter[index++] = 100000;

    // load data to GPU
    printf("# Begin loading to GPU\n");
    pq_float *device_codeword_pq, *device_codeword_nq, *device_query, *device_lookup_table;
    int *device_ret_result, *device_prefixsum_result, *device_mid_result;
    pq_int *device_codebook, *device_q_map;
    cudaMalloc((pq_float **)&device_codeword_nq, size_codeword_nq * sizeof(pq_float));
    cudaMalloc((pq_float **)&device_codeword_pq, size_codeword_pq * sizeof(pq_float));
    cudaMalloc((pq_int **)&device_codebook, size_codebook * sizeof(pq_int));
    cudaMalloc((pq_float **)&device_query, size_device_query * sizeof(pq_float));
    cudaMalloc((pq_int **)&device_q_map, size_q_map * sizeof(pq_int));
    cudaMalloc((pq_float **)&device_lookup_table, size_lookup_table * sizeof(pq_float));
    cudaMalloc((int **)&device_mid_result, batch_vecs * sizeof(int));
    cudaMalloc((int **)&device_prefixsum_result, batch_vecs * sizeof(int));
    cudaMalloc((int **)&device_ret_result, size_ret_result * sizeof(int));

    int *h_mid_result = new int[batch_vecs];
    int *h_prefixsum_result = new int[batch_vecs];
    int *h_ret_result = new int[batch_vecs];
    pq_float *h_lookup_table = new pq_float[size_lookup_table];
    pq_float *gpuref_lookup_table = new pq_float[size_lookup_table];
    int *gpuref_mid_result = new int[batch_vecs];

    printf("# Begin copying data\n");

    cudaMemcpy(device_codeword_nq, codeword_nq, size_codeword_nq * sizeof(pq_float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_codeword_pq, codeword_pq, size_codeword_pq * sizeof(pq_float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_codebook, codebook, size_codebook * sizeof(pq_int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_q_map, q_map, size_q_map * sizeof(pq_int), cudaMemcpyHostToDevice);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, device_mid_result, device_prefixsum_result, batch_vecs);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // test lookup table
    // cudaMemcpy(device_query, query, num_dimen * sizeof(pq_float), cudaMemcpyHostToDevice);
    // dim3 grid_lookup (num_q);
    // dim3 block_lookup (Ks);
    // calLookupOnGPU<<<grid_lookup, block_lookup>>>(device_query, device_codeword_nq, device_codeword_pq, device_q_map, device_lookup_table,
    //     num_dimen);
    // cudaDeviceSynchronize();
    // cudaMemcpy(gpuref_lookup_table, device_lookup_table, size_lookup_table * sizeof(pq_float), cudaMemcpyDeviceToHost);
    // calLookupOnCPU(query, codeword_nq, codeword_pq, q_map, h_lookup_table, num_dimen, num_q, Ks);
    // checkLookup(h_lookup_table, gpuref_lookup_table, size_lookup_table);

    // int start_pos = 0, end_pos = batch_vecs;
    // dim3 grid_prune (batch_vecs / threads_per_block);
    // dim3 block_prune (threads_per_block);
    // calApproxVecs<<<grid_prune, block_prune, size_lookup_table * sizeof(pq_float)>>>(device_mid_result, device_codebook, device_lookup_table, device_q_map, 
    //     num_q, Ks, num_vecs, start_pos, end_pos, threshold[0]);
    // cudaDeviceSynchronize();
    // cudaMemcpy(gpuref_mid_result, device_mid_result, batch_vecs * sizeof(pq_float), cudaMemcpyDeviceToHost);
    // calMidResultOnCPU(h_mid_result, codebook, h_lookup_table, q_map, num_q, Ks, num_vecs, start_pos, end_pos, threshold[0]);
    // checkMidResult(h_mid_result, gpuref_mid_result, end_pos - start_pos);
    
    // end testing lookup table

    // int sum_batch_query = (num_query + batch_query - 1) / batch_query;
    int sum_batch_vecs = (num_vecs - top_norm + batch_vecs - 1) / batch_vecs;
    // vector<int> candidate;

    int sum_final_correct = 0, temp_length[2];
    long long sum_final_length = 0;
    long long sum_filter_length = 0;

    int batch_print = num_query / 10;
    double cpu_time = 0;
    clock_t start_time_cpu = 0;
    clock_t end_time_cpu = 0;
    clock_t start_time = clock();

    printf("# Begin calculating\n");
    for (int i = 0; i < num_query; ++i) {
        // load the query into gpu
        cudaMemcpy(device_query, query + i * num_dimen, num_dimen * sizeof(pq_float), cudaMemcpyHostToDevice);
        // calculate the lookup table
        dim3 grid_lookup (num_q);
        dim3 block_lookup (Ks);

        calLookupOnGPU<<<grid_lookup, block_lookup>>>(device_query, device_codeword_nq, device_codeword_pq, device_q_map, device_lookup_table,
            num_dimen);
        // calculate the approximate vecs
        dim3 grid_prune (batch_vecs / threads_per_block);
        dim3 block_prune (threads_per_block);

        int start_pos = top_norm, end_pos = top_norm + batch_vecs, current_length = 0;
        float norm_threshold = threshold[i] / query_norm[i];

        // printf("# %dth query threshold is %f", i, norm_threshold);

        // dim3 grid_ret (size_ret_result + threads_per_block - 1 / threads_per_block);
        // dim3 block_ret (threads_per_block);

        // initResult<<<grid_ret, block_ret>>>(device_ret_result, size_ret_result);

        cudaDeviceSynchronize();

        // if (i == 900) {
        //     printf("# %d\n", i);
        // }

        for (int j = 0; j < sum_batch_vecs; ++j) {
            if (norm_threshold > norm_filter[j]) {
                sum_filter_length += num_vecs - top_norm - j * batch_vecs;
                break;
            }
            calApproxVecs<<<grid_prune, block_prune, size_lookup_table * sizeof(pq_float)>>>(device_mid_result, device_codebook, device_lookup_table, 
                device_q_map, num_q, Ks, num_vecs, start_pos, end_pos, threshold[i]);
            cudaDeviceSynchronize();
            // cudaMemcpy(h_mid_result, device_mid_result, batch_vecs * sizeof(int), cudaMemcpyDeviceToHost);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, device_mid_result, device_prefixsum_result, batch_vecs);
            cudaDeviceSynchronize();
            // cudaMemcpy(h_prefixsum_result, device_prefixsum_result, batch_vecs * sizeof(int), cudaMemcpyDeviceToHost);
            assignResult<<<grid_prune, block_prune>>>(device_prefixsum_result, device_ret_result, start_pos, end_pos, current_length);
            cudaDeviceSynchronize();
            cudaMemcpy(temp_length, device_prefixsum_result + end_pos - start_pos - 1, sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(h_ret_result, device_prefixsum_result, batch_vecs * sizeof(int), cudaMemcpyDeviceToHost);
            current_length += temp_length[0];
            start_pos += batch_vecs;
            end_pos += batch_vecs;
            if (end_pos > num_vecs) end_pos = num_vecs;
        } 
        cudaMemcpy(ret_result, device_ret_result, current_length * sizeof(int), cudaMemcpyDeviceToHost);
        start_time_cpu = clock();
        for (int j = 0; j < topk; ++j) ret_result[current_length + j] = candidate_init[i][j];

        // bool flag = 0;
        // for (int j = 0; j < current_length; ++j) {
        //     if (ret_result[j] >= num_vecs) {
        //         flag = 1;
        //         break;
        //     }
        // }

        // if (flag) {
        //     printf("# %dth query outputs wrong with threshold %f!\n", i, threshold[i]);
        //     for (int j = 0; j < current_length; ++j) {
        //         printf("%d ", ret_result[j]);
        //     }
        //     printf("\n");
        //     break;
        // }

        current_length += topk;
        sum_final_length += current_length;
        query_idx = i;
        std::sort(ret_result, ret_result + current_length, cmpIPIndex);
        sum_final_correct += sameIndex(ret_result, answer[i], topk);
        end_time_cpu = clock();
        cpu_time += (double)(end_time_cpu - start_time_cpu) / CLOCKS_PER_SEC;
        if (i % batch_print == 0) printf("# %dth query has been processed\n", i);
    }

    clock_t end_time = clock();
    printf("\n# time spend: %fs\n# top norm faiss time spend: %fs\n# total query: %d\n# recall: %f\n# norm filter length: %f\n# cpu time: %fs", 
        (double)(end_time - start_time) / CLOCKS_PER_SEC + total_time, total_time, num_query, 
        sum_final_correct * 1.0 / num_query / topk, sum_filter_length * 1.0 / num_query, cpu_time);
    printf("\n# init recall: %f\n# total vecs: %d\n# final_length: %f\n", sum_init_correct * 1.0 / num_query / topk, num_vecs, 
        sum_final_length * 1.0 / num_query);    

    // delete all the data
    cudaFree(device_codeword_nq);
    cudaFree(device_codeword_pq);
    cudaFree(device_query);
    cudaFree(device_codebook);
    cudaFree(device_q_map);
    cudaFree(device_lookup_table);
    cudaFree(device_mid_result);
    cudaFree(device_prefixsum_result);
    cudaFree(device_ret_result);
    cudaFree(d_temp_storage);

    delete [] codeword_nq;
    delete [] codeword_pq;
    delete [] codebook;
    delete [] query;
    delete [] query_norm;
    delete [] threshold;
    delete [] vecs;
    delete [] norm_filter;
    delete [] ret_result;
    delete [] h_mid_result;
    delete [] h_prefixsum_result;
    delete [] h_ret_result;
    for (int i = 0; i < num_query; ++i) delete [] candidate_init[i];
    delete [] candidate_init;
    for (int i = 0; i < num_query; ++i) delete [] answer[i];
    delete [] answer;

    return 0;
}