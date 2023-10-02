#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>
#include <omp.h>

#include "utils.hpp"
#include <cmath>
#include <cstring>

#define MASTER 0
#define TAG_GATHER 0
#define TAG_HALO 1

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    const char* input_filepath = "/home/george/src/CSC4005-2023Fall/project1/images/20K-RGB-Benchmark-Image.jpg";
    const char* output_filepath = "/home/george/src/CSC4005-2023Fall/project1/images/20K-RGB-Benchmark-Image-Smoothed-mpi.jpg";

    MPI_Init(&argc, &argv);
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Request requests[4]; // two for send and two for recv operations

    auto input_jpeg = read_from_jpeg(input_filepath);
    auto start_time = std::chrono::high_resolution_clock::now();

    int total_pixel_num = input_jpeg.width * input_jpeg.height;
    int pixel_num_per_task = total_pixel_num / numtasks;
    int left_pixel_num = total_pixel_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_pixel_num = 0;
    for (int i = 0; i < numtasks; i++) {
        if (divided_left_pixel_num < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            divided_left_pixel_num++;
        } else {
            cuts[i+1] = cuts[i] + pixel_num_per_task;
        }
    }

    auto local_input = new unsigned char[(cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels];
    auto local_output = new unsigned char[(cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels];
    std::memcpy(local_input, input_jpeg.buffer + cuts[taskid] * input_jpeg.num_channels, (cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels);

    int num_requests = 0;

    // Initiate non-blocking halo exchanges
    if (taskid != 0) {
        MPI_Irecv(local_input, input_jpeg.width * input_jpeg.num_channels, MPI_CHAR, taskid - 1, TAG_HALO, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if (taskid != numtasks - 1) {
        MPI_Isend(local_input + ((cuts[taskid + 1] - cuts[taskid]) - input_jpeg.width) * input_jpeg.num_channels, input_jpeg.width * input_jpeg.num_channels, MPI_CHAR, taskid + 1, TAG_HALO, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Smoothing using OpenMP to parallelize
    #pragma omp parallel for
    for (int idx = 0; idx < (cuts[taskid + 1] - cuts[taskid]); idx++) {
        int height = idx / input_jpeg.width;
        int width = idx % input_jpeg.width;

        if (height == 0 || width == 0 || height == input_jpeg.height - 1 || width == input_jpeg.width - 1) {
            for (int channel = 0; channel < input_jpeg.num_channels; channel++) {
                local_output[idx * input_jpeg.num_channels + channel] = local_input[idx * input_jpeg.num_channels + channel];
            }
        } else {
            for (int channel = 0; channel < input_jpeg.num_channels; channel++) {
                double sum = 0.0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        sum += local_input[((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels + channel] * filter[i + 1][j + 1];
                    }
                }
                local_output[idx * input_jpeg.num_channels + channel] = static_cast<unsigned char>(std::round(sum));
            }
        }
    }

    // Wait for non-blocking operations to complete
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    if (taskid == MASTER) {
        std::memcpy(input_jpeg.buffer, local_output, (cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels);
        for (int i = 1; i < numtasks; i++) {
            MPI_Recv(input_jpeg.buffer + cuts[i] * input_jpeg.num_channels, (cuts[i + 1] - cuts[i]) * input_jpeg.num_channels, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        JPEGMeta output_jpeg{input_jpeg.buffer, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};

        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        std::cout << "Smoothing Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } else {
        MPI_Send(local_output, (cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    delete[] local_input;
    delete[] local_output;
    MPI_Finalize();
    return 0;
}
