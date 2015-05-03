// ImageNormalization.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
//
//
//int _tmain(int argc, _TCHAR* argv[])
//{
//	return 0;
//}

#include <opencv2/core/core.hpp>      // Basic OpenCV structures
#include <opencv2/imgproc/imgproc.hpp>// Image processing methods for the CPU
#include <opencv2/highgui/highgui.hpp>// Read images
#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image;
    gpu::GpuMat g_image;
    gpu::GpuMat g_out;

    image = imread(argv[1], IMREAD_GRAYSCALE); // Read the file
    Mat image_orig;
    image.copyTo(image_orig);

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    g_image.upload(image);

    Ptr<gpu::CLAHE> clahe = gpu::createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(Size(16, 16));
    clahe->apply(g_image, g_out);
    g_out.download(image);

    namedWindow("Equalized", WINDOW_NORMAL); // Create a window for display.
    imshow("Equalized", image); // Show our image inside it.

    namedWindow("Original", WINDOW_NORMAL); // Create a window for display.
    imshow("Original", image_orig); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}