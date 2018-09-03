#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap_image.hpp"
#include <chrono>
#include <omp.h>

using namespace std;

void mygetmatch(float *I, float *T, int Iwidth, int Iheight, int Twidth, int Theight) {
	int difference = 0;
	int counter = 0;
	//create a new team of threads
#pragma omp parallel 
	//divides the work amongst the threads of the currently executing team
#pragma omp  for schedule(dynamic)
	for (int ii = 0; ii < Iheight; ii++) {
		//#pragma omp parallel for 
		for (int ij = 0; ij < Iwidth; ij++)
		{
			for (int ti = 0; ti < Theight; ti++)
			{
				for (int tj = 0; tj < Twidth; tj++) {
					difference = I[ii * Iwidth + ij] - T[ti * Twidth + tj];
					if (difference != 0)
					{
						break;
					}
					else
						if (ij < Iwidth)
							ij++;

					if (ti == Theight - 1 && tj == Twidth - 1)
					{
						//find
						counter++;
					}
				}
			}

		}
	}
	printf("\n%d found\n", counter);
}

void rotate(float *in, int width, int height, float *out) {
	float *b;
	int index = 0;
	b = (float *)malloc(sizeof(float) * width*height);
	//create a new team of threads
#pragma omp parallel for num_threads(1000) 
	for (int i = 0; i<width; i++)
	{
		for (int j = height - 1; j >= 0; j--)
		{
			out[index] = in[j*width + i];
			index++;
		}
	}
}

float *read(const char *bmpName, int *width, int *height) {
	std::string file_name(bmpName);

	bitmap_image image(file_name);
	if (!image)
	{
		printf("Error - Failed to open '%s'\n", file_name.c_str());
		return 0;
	}
	int h = image.height();
	int w = image.width();
	int pixels = h*w;
	float *output;
	rgb_t colour;
	output = (float *)malloc(sizeof(float) * pixels);
#pragma omp parallel for 
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			image.get_pixel(j, i, colour);
			output[i*w + j] = colour.red + colour.green + colour.blue;
			//printf("[%d]", output[i*w + j]);
		}

	}
	*width = w;
	*height = h;
	return output;

}

void omp_check() {
	//omp_set_nested(2);
	printf("------------ Info -------------\n");
#ifdef _DEBUG
	printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
	printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
	printf("[-] Platform: x64\n");
#elif _M_IX86 
	printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
	printf("[-] OpenMP is on.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
	printf("[!] OpenMP is off.\n");
	printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
	printf("[-] Maximum threads: %d\n", omp_get_max_threads());
	printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}


int main(int argc, char *argv[]) {
	omp_check();
	const auto max_num_threads = omp_get_max_threads();
	printf("%d", max_num_threads);
	omp_set_num_threads(4);
	for (int i = 1; i < argc; i += 2) {
		int I_width, I_height, T_width, T_height;
		float *I, *T;

		if (argc % 2 == 0) {
			printf("Usage: template-matching-cpu original.bmp template.bmp\n");
			exit(0);
		}
		// capture start time
		auto start_time = chrono::high_resolution_clock::now();

#pragma omp parallel sections
		{
#pragma omp section
		{
			I = read(argv[i], &I_width, &I_height);
		}
#pragma omp section
		{
			T = read(argv[i + 1], &T_width, &T_height);
		}
		}

		if (I == 0 || T == 0) {
			exit(1);
		}

		if (I_width < T_width || I_height < T_height) {
			fprintf(stderr, "Error: The template is larger than the picture\n");
			exit(EXIT_FAILURE);
		}

		int w2 = T_height;
		int h2 = T_width;
		float *output1;

		int w3 = h2;
		int h3 = w2;
		float *output2;

		int w4 = h3;
		int h4 = w3;
		float *output3;

#pragma omp parallel sections
		{
#pragma omp section
		{
			mygetmatch(I, T, I_width, I_height, T_width, T_height);
		}
#pragma omp section
		{
			//second round

			output1 = (float *)malloc(sizeof(float) * T_width*T_height);
			rotate(T, T_width, T_height, output1);
		}
		}



#pragma omp parallel sections
		{
#pragma omp section
		{
			mygetmatch(I, output1, I_width, I_height, w2, h2);

		}
#pragma omp section
		{
			//third round

			output2 = (float *)malloc(sizeof(float) * T_width*T_height);
			rotate(output1, w2, h2, output2);
		}
		}

#pragma omp parallel sections
		{
#pragma omp section
		{
			mygetmatch(I, output2, I_width, I_height, w3, h3);
		}
#pragma omp section
		{
			//forth round
			output3 = (float *)malloc(sizeof(float) * T_width*T_height);
			rotate(output2, w2, h2, output3);
		}
		}
		mygetmatch(I, output3, I_width, I_height, w4, h4);



		// capture end time
		auto end_time = chrono::high_resolution_clock::now();
		// elapsed time in milliseconds
		auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
		cout << "CPU duration: " << duration.count() << "ms" << endl;
	}
}