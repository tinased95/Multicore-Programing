#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap_image.hpp"
#include <chrono>

using namespace std;

int mygetmatch(float *I, float *T, int Iwidth, int Iheight, int Twidth, int Theight) {
	int difference;
	int counter = 0;
	for (int ii = 0; ii < Iheight; ii++) {
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
	//printf("\n%d found\n", counter);
	return counter;
}

void rotate(float *in, int width, int height, float *out) {
	float *b;
	int index = 0;
	b = (float *)malloc(sizeof(float) * width*height);
	//printf("b:\n");
	for (int i = 0; i<width; i++)
	{
		for (int j = height - 1; j >= 0; j--)
		{
			out[index] = in[j*width + i];
			index++;
			//printf("%f  ", in[j*width + i]);
		}
		//printf("\n");
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

int main(int argc, char *argv[]) {
	int counter;
	for (int i = 1; i < argc; i += 2) {
		counter = 0;
		int I_width, I_height, T_width, T_height;
		float *I, *T;
		int x1, y1, x2, y2;

		if (argc % 2 == 0) {
			printf("Usage: template-matching-cpu original.bmp template.bmp\n");
			exit(0);
		}
		// capture start time
		auto start_time = chrono::high_resolution_clock::now();
		I = read(argv[1], &I_width, &I_height);
		T = read(argv[2], &T_width, &T_height);

		if (I == 0 || T == 0) {
			exit(1);
		}

		if (I_width < T_width || I_height < T_height) {
			fprintf(stderr, "Error: The template is larger than the picture\n");
			exit(EXIT_FAILURE);
		}
		counter += mygetmatch(I, T, I_width, I_height, T_width, T_height);

		//second round
		int w2 = T_height;
		int h2 = T_width;
		float *output1;
		output1 = (float *)malloc(sizeof(float) * T_width*T_height);
		rotate(T, T_width, T_height, output1);
		counter += mygetmatch(I, output1, I_width, I_height, w2, h2);


		//third round
		int w3 = h2;
		int h3 = w2;
		float *output2;
		output2 = (float *)malloc(sizeof(float) * T_width*T_height);
		rotate(output1, w2, h2, output2);
		counter += mygetmatch(I, output2, I_width, I_height, w3, h3);

		//forth round
		int w4 = h3;
		int h4 = w3;
		float *output3;
		output3 = (float *)malloc(sizeof(float) * T_width*T_height);
		rotate(output2, w2, h2, output3);
		counter += mygetmatch(I, output3, I_width, I_height, w4, h4);



		// capture end time
		auto end_time = chrono::high_resolution_clock::now();
		// elapsed time in milliseconds
		auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
		//cout << "CPU duration: " << duration.count() << "ms" << endl;
		printf("%d\n", counter);
	}
}