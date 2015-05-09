// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <algorithm>

#include <opencv2/core/core.hpp>      // Basic OpenCV structures
#include <opencv2/imgproc/imgproc.hpp>// Image processing methods for the CPU
#include <opencv2/highgui/highgui.hpp>// Read images
#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods

using namespace cv;
using namespace std;
#include "Stopwatch.hpp"
