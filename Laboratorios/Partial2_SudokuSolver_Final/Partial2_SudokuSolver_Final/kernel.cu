/*Paralized part of the code based on the code made by: SergioMEV
https://github.com/SergioMEV/sudoku_solver/blob/main/sudoku.cu
Author: Eduardo Gonzalez
CoAuthor: SergioMEV
*/

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define BOARD_DIM 9
#define GROUP_DIM 3

//------------------------------------------Start CPU----------------------------------------------//
void printMatrix(int matrix[9][9]) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << "-----------------------" << endl;
}

bool isValid(int grid[9][9], int r, int c, int k) {
    // Verificar fila
    for (int i = 0; i < 9; ++i) {
        if (grid[r][i] == k) {
            return false;
        }
    }

    // Verificar columna
    for (int i = 0; i < 9; ++i) {
        if (grid[i][c] == k) {
            return false;
        }
    }

    // Verificar caja
    int boxRow = r / 3 * 3;
    int boxCol = c / 3 * 3;
    for (int i = boxRow; i < boxRow + 3; ++i) {
        for (int j = boxCol; j < boxCol + 3; ++j) {
            if (grid[i][j] == k) {
                return false;
            }
        }
    }

    return true;
}

bool solve(int grid[9][9], int r = 0, int c = 0) {

    //printMatrix(grid);
    if (r == 9) {
        return true;
    }
    else if (c == 9) {
        return solve(grid, r + 1, 0);
    }
    else if (grid[r][c] != 0) {
        return solve(grid, r, c + 1);
    }
    else {
        for (int k = 1; k <= 9; ++k) {
            if (isValid(grid, r, c, k)) {
                grid[r][c] = k;
                if (solve(grid, r, c + 1)) {
                    return true;
                }
                grid[r][c] = 0;
            }
        }
        return false;
    }
}
//------------------------------------------End   CPU----------------------------------------------//

//------------------------------------------Start GPU----------------------------------------------//
typedef struct board {
    uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;

__host__ __device__ uint16_t digit_to_cell(int digit);
__host__ __device__ int cell_to_digit(uint16_t cell);

__global__ void cell_solver(board_t* boards) {
    size_t cell_idx = threadIdx.x;
    uint16_t current_cell;
    size_t votes;

    __shared__ board_t board;
    board.cells[cell_idx] = boards[blockIdx.x].cells[cell_idx];
    __syncthreads();

    do {
        current_cell = board.cells[cell_idx];
        if (cell_to_digit(current_cell) != 0) break;

        size_t col_idx = cell_idx % 9;
        for (size_t index = col_idx; index < col_idx + 9 * 9; index += 9) {
            if (index == cell_idx) continue;
            int digit_result = cell_to_digit(board.cells[index]);
            if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
        }

        if (cell_to_digit(current_cell) != 0) break;

        size_t start_idx = cell_idx - col_idx;
        for (size_t index = start_idx; index < start_idx + 9; index++) {
            if (index == cell_idx) continue;
            int digit_result = cell_to_digit(board.cells[index]);
            if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
        }

        if (cell_to_digit(current_cell) != 0) break;

        size_t reduced_index = cell_idx - (cell_idx / 27) * 27;
        size_t minor_row = reduced_index / 9;
        size_t minor_col = (reduced_index - minor_row * 9) % 3;
        size_t start_index = cell_idx - minor_col - minor_row * 9;

        for (size_t row = 0; row < 3; row++) {
            for (size_t col = 0; col < 3; col++) {
                size_t index = start_index + col + row * 9;
                if (index == cell_idx) continue;
                int digit_result = cell_to_digit(board.cells[index]);
                if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
            }
        }

        votes = __syncthreads_count(board.cells[cell_idx] != current_cell);

    } while (votes != 0);

    boards[blockIdx.x].cells[cell_idx] = board.cells[cell_idx];
}

void solve_boards(board_t* cpu_boards, size_t num_boards) {
    board_t* gpu_boards;
    if (cudaMalloc(&gpu_boards, sizeof(board_t) * num_boards) != cudaSuccess) {
        perror("cuda malloc failed.");
        exit(2);
    }

    if (cudaMemcpy(gpu_boards, cpu_boards, sizeof(board_t) * num_boards, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        perror("cuda memcpy failed. ");
        exit(2);
    }

    cell_solver << <1, 81 >> > (gpu_boards);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        perror("Synchronized failed.");
        exit(2);
    }

    if (cudaMemcpy(cpu_boards, gpu_boards, sizeof(board_t) * num_boards, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        perror("cuda memcpy failed. ");
        exit(2);
    }
}

__host__ __device__ int custom_ctz(uint16_t value) {
    int count = 0;
    while ((value & 1) == 0 && count < sizeof(uint16_t) * 8) {
        value >>= 1;
        count++;
    }
    return count;
}

__host__ __device__ uint16_t digit_to_cell(int digit) {
    if (digit == 0) {
        return 0x3FE; // All digits from 1 to 9 are possible
    }
    else {
        return 1 << digit; // Encode the digit
    }
}

__host__ __device__ int cell_to_digit(uint16_t cell) {
    int lsb = custom_ctz(cell);
    if (cell == 1 << lsb)
        return lsb; // Only one possibility, return the digit
    else
        return 0; // Multiple possibilities or no possibilities
}

void print_board(board_t* board) {
    printf("+-------+-------+-------+\n");
    for (int i = 0; i < BOARD_DIM; i++) {
        if (i > 0 && i % GROUP_DIM == 0) {
            printf("|-------+-------+-------|\n");
        }
        for (int j = 0; j < BOARD_DIM; j++) {
            if (j > 0 && j % GROUP_DIM == 0) {
                printf("| ");
            }
            int digit = cell_to_digit(board->cells[i * BOARD_DIM + j]);
            if (digit == 0) {
                printf(". ");
            }
            else {
                printf("%d ", digit);
            }
        }
        printf("|\n");
    }
    printf("+-------+-------+-------+\n");
}
//------------------------------------------End   GPU----------------------------------------------//

int main(int argc, char** argv) {

    board_t boards[1];

    int matrizCPU[9][9] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    int matrizGPU[81] = {
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    };

    /*-----------------------CPU-----------------------------------------------------*/
    cout << "Original: \n" << endl;
    printMatrix(matrizCPU);
    cout << endl;

    cout << "CPU: \n" << endl;
    auto start = chrono::high_resolution_clock::now();
    bool jala = solve(matrizCPU, 0, 0);
    auto end = chrono::high_resolution_clock::now();
    cout << "Solucion CPU: " << endl;
    printMatrix(matrizCPU);

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Imprimir el tiempo de ejecución en milisegundos
    cout << "Tiempo de ejecución en CPU: " << duration.count() << " microseconds" << endl;

    /*-----------------------GPU-----------------------------------------------------*/
    cout << "GPU: \n" << endl;
    //Rellenamos el board
    board_t board;
    for (int i = 0; i < BOARD_DIM * BOARD_DIM; i++) {
        board.cells[i] = matrizGPU[i] == 0 ? 0x3FE : 1 << matrizGPU[i];
    }

    boards[0] = board;

    //Ejecucion del codigo
    auto start2 = chrono::high_resolution_clock::now();
    solve_boards(boards, 1);
    auto end2 = chrono::high_resolution_clock::now();
    print_board(boards);

    auto duration2 = chrono::duration_cast<chrono::microseconds>(end2 - start2);

    // Imprimir el tiempo de ejecución en milisegundos
    cout << "Tiempo de ejecución en GPU: " << duration2.count() << " microseconds" << endl;


    return 0;
}
