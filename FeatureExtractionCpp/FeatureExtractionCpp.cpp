// FeatureExtractionCpp.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "HaemoragingImage.hpp"
#include "HistogramNormalize.hpp"

namespace fs = boost::filesystem;

const char* keys =
{
    "{ref||| must specify reference image name}"
    "{target||| must specify image name}"
    "{in|inputDir||directory to read files from}"
    "{out|outDir||output directory}"
    "{d|debug||invoke debugging functionality}"

};

Mat src, src_gray, reference;
RNG rng(12345);
string sourceWindow("Source");
string targetWindow("Target");
string transformedWindow("Transformed");
string enhancedWindow("Enhanced");

ParamBag params;
unique_ptr<HaemoragingImage> haemorage(new HaemoragingImage);

vector<vector<Point>> * FindHaemorages(unique_ptr<HaemoragingImage>&, Mat&, ParamBag&);

void thresh_callback(int, void *)
{
    unique_ptr<vector<vector<Point>>> contours(FindHaemorages(haemorage, src, params));

    Mat img;
    src.copyTo(img);
    TransformImage::DrawContours(*contours, vector<Vec4i>(), img);

    /// Draw contours
    /// Show in a window
    imshow(sourceWindow, img);
}

// color transfer experiments
void do_debug(CommandLineParser& parser)
{
    // keep for debugging
    string ref_file_name = parser.get<string>("ref");
    string file_name = parser.get<string>("target");

    // debugging stuff
    Mat rgb;
    rgb = imread(file_name, IMREAD_COLOR);

    auto hi = HaemoragingImage(rgb);
    hi.PyramidDown();
    src = hi.getEnhanced();

    rgb = imread(ref_file_name, IMREAD_COLOR);

    auto ref_image = HaemoragingImage(rgb);
    ref_image.PyramidDown();
    reference = ref_image.getEnhanced();

    Channels _channels[3] = { Channels::RED, Channels::GREEN, Channels::BLUE };
    vector<Channels> channels(_channels, _channels + 3);

    auto histSpec = HistogramNormalize(reference, channels);

    Mat dest;
    histSpec.HistogramSpecification(src, dest);

    //3. CLAHE
    hi.setImage(dest);
    hi.MakeHsv();
    hi.GetOneChannelImages(Channels::V);
    hi.ApplyClahe();

    dest = hi.getEnhanced();

    cvtColor(dest, rgb, COLOR_HSV2BGR);

    namedWindow(sourceWindow, WINDOW_NORMAL);
    namedWindow(targetWindow, WINDOW_NORMAL);
    namedWindow(transformedWindow, WINDOW_NORMAL);

    imshow(sourceWindow, reference);
    imshow(targetWindow, src);
    imshow(transformedWindow, dest);
    imshow(enhancedWindow, rgb);

    //params.cannyThresh = 30;
    //createTrackbar("Track", sourceWindow, &(params.cannyThresh), 100, thresh_callback);
    //thresh_callback(0, &(params.cannyThresh));
    //ref_image.DisplayEnhanced(true);
    waitKey(0);

}

//1. Pyramid Down
//2. Histogram specification: 6535_left
//3. Histogram equalization (CLAHE) on V channel of the HSV image
//4. Resize to 100x100
//5. Write to out_path
void process_files(string& ref, fs::path& in_path, vector<string>& in_files, fs::path& out_path)
{
    params.cannyThresh = 30;
    params.blockSize = 11;

    // process reference image
    Mat rgb = imread(ref, IMREAD_COLOR);

    auto ref_image = HaemoragingImage(rgb);
    ref_image.PyramidDown();
    Mat reference = ref_image.getEnhanced();

    Channels _channels[3] = { Channels::RED, Channels::GREEN, Channels::BLUE };
    vector<Channels> channels(_channels, _channels + 3);
    
    // create the class for histogram specification
    auto histSpec = HistogramNormalize(reference, channels);


    for (string& in_file : in_files)
    {
        // in-path
        fs::path filePath = in_path /= fs::path(in_file);
        // out-path
        fs::path outFilePath = out_path /= fs::path(in_file);

        // read it
        rgb = imread(filePath.string(), IMREAD_COLOR);
        HaemoragingImage hi(rgb);
        // 1. Pyramid down
        hi.PyramidDown();
        src = hi.getEnhanced();

        // 2. Histogram specification
        Mat dest;
        histSpec.HistogramSpecification(src, dest);

        //3. CLAHE
        hi.setImage(dest);
        hi.MakeHsv();
        hi.GetOneChannelImages(Channels::V);
        hi.ApplyClahe();
        
        src = hi.getEnhanced();

        //4. resize
        resize(src, dest, Size(100, 100));
        cvtColor(dest, rgb, COLOR_HSV2BGR);

        //5. write out
        imwrite(out_path.string(), rgb);
    }
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    string in_dir = parser.get<string>("in");
    string out_dir = parser.get<string>("out");
    string ref = parser.get<string>("ref");
    
    bool debug = parser.get<bool>("debug");

    if (debug)
    {
        do_debug(parser);
        return 0;
    }

    fs::path in_path(in_dir);
    fs::path out_path(out_dir);

    if (fs::exists(out_path))
    {
        fs::remove_all(out_path);
    }

    fs::create_directories(out_path);

    // print out GPU information
    gpu::printCudaDeviceInfo(0);
    fs::directory_iterator it(in_path), enumer(in_path);

    vector<string> in_files;
    
    DIR *dir;
    struct dirent *ent;

    // using dirent because it actually finishes (unlike boost)
    dir = opendir(in_dir.c_str());
    if (dir != NULL)
    {
        /* Print all files and directories within the directory */
        while ((ent = readdir(dir)) != NULL)
        {
            if (ent->d_type == DT_REG)
            {
                in_files.push_back(ent->d_name);
            }
        }
    }
    closedir(dir);

    process_files(ref, in_path, in_files, out_path);
    return(0);
}

