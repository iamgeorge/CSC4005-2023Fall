//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of image smoothing
//

#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>    // MPI Header

#include "utils.hpp"
#include <cmath>


#define MASTER 0
#define TAG_GATHER 0

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    // Verify input argument format
    argv[1] = const_cast<char*>("/home/george/src/CSC4005-2023Fall/project1/images/Lena-RGB.jpg");
    argv[2] = const_cast<char*>("/home/george/src/CSC4005-2023Fall/project1/images/Lena-Smooth.jpg");
    argc = 3;

    // Start the MPI
    MPI_Init(&argc, &argv);
    int numtasks, taskid, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Status status;

    // Read JPEG File
    const char* input_filepath = argv[1];
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
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }

    auto smoothed_image = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    for (int idx = cuts[taskid]; idx < cuts[taskid + 1]; idx++) {
        int height = idx / input_jpeg.width;
        int width = idx % input_jpeg.width;
        if (height == 0 || width == 0 || height == input_jpeg.height - 1 || width == input_jpeg.width - 1) {
            // Copy the border pixels directly
            for (int channel = 0; channel < input_jpeg.num_channels; channel++) {
                smoothed_image[idx * input_jpeg.num_channels + channel] = input_jpeg.buffer[idx * input_jpeg.num_channels + channel];
            }
        } else {
            for (int channel = 0; channel < input_jpeg.num_channels; channel++) {
                double sum = 0.0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        sum += input_jpeg.buffer[((height + i) * input_jpeg.width + (width + j)) * input_jpeg.num_channels + channel] * filter[i + 1][j + 1];
                    }
                }
                smoothed_image[idx * input_jpeg.num_channels + channel] = static_cast<unsigned char>(std::round(sum));
            }
        }
    }

    if (taskid == MASTER) {
        for (int i = MASTER + 1; i < numtasks; i++) {
            MPI_Recv(smoothed_image + cuts[i] * input_jpeg.num_channels, (cuts[i + 1] - cuts[i]) * input_jpeg.num_channels, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Save the Smoothed Image
        const char* output_filepath = argv[2];
        JPEGMeta output_jpeg{smoothed_image, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        std::cout << "Smoothing Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    } else {
        MPI_Send(smoothed_image + cuts[taskid] * input_jpeg.num_channels, (cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    delete[] smoothed_image;
    MPI_Finalize();
    return 0;
}
