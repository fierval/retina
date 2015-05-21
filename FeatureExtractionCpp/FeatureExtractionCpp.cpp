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
    "{size||128|output image dimensions}"
    "{d|debug||invoke debugging functionality}"
    "{t|threshold|12|Canny threshold}"

};

Mat src;
RNG rng(12345);
string sourceWindow("Reference");
string targetWindow("Target");
string transformedWindow("Transformed");
string enhancedWindow("Enhanced");
string contouredWindow("Contoured");

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
    int dim = parser.get<int>("size");
    int thresh = parser.get<int>("t");

    Size size(dim, dim);

    // debugging stuff
    // load the target image & show it
    Mat rgb;
    rgb = imread(file_name, IMREAD_COLOR);

    // create the image mask to be used last
    auto hi = HaemoragingImage(rgb);
    hi.PyramidDown();
    hi.getEnhanced().copyTo(src);

    hi.setImage(src);
    int i = 0;
    do
    {
        hi.CreateEyeContours(i == 0 ? thresh : 1);
        auto mask = hi.CreateMask(dim);
        i++;

        cout << "Eye area: " << hi.EyeAreaRatio() << endl;
        namedWindow("mask");
        imshow("mask", mask);
    } while (hi.EyeAreaRatio() < 0.48 && i <= 2);


    hi.DrawEyeContours(src, Scalar(0, 0, 255), 2);
    namedWindow(targetWindow, WINDOW_NORMAL);
    imshow(targetWindow, src);

    // load the reference image & show it
    rgb = imread(ref_file_name, IMREAD_COLOR);

    // show image contours and filter its background
    HaemoragingImage ref_haem(rgb);
    ref_haem.PyramidDown();
    Mat reference(ref_haem.getEnhanced());

    namedWindow(sourceWindow, WINDOW_NORMAL);
    imshow(sourceWindow, reference);

    Channels _channels[3] = { Channels::RED, Channels::GREEN, Channels::BLUE };
    vector<Channels> channels(_channels, _channels + 3);

    auto histSpec = HistogramNormalize(reference, channels);

    Mat dest;
    histSpec.HistogramSpecification(src, dest);
    namedWindow(transformedWindow, WINDOW_NORMAL);
    imshow(transformedWindow, dest);

    //3. CLAHE
    hi.setImage(dest);
    hi.MakeHsv();
    hi.GetOneChannelImages(Channels::V);
    hi.ApplyClahe();

    dest = hi.getEnhanced();

    cvtColor(dest, rgb, COLOR_HSV2BGR);
    Mat sized;
    resize(rgb, sized, size);

    //apply background filtering mask
    hi.setImage(sized);
    hi.MaskOffBackground();

    namedWindow(enhancedWindow, WINDOW_NORMAL);
    imshow(enhancedWindow, hi.getEnhanced());

    params.cannyThresh = 60;
    //createTrackbar("Track", sourceWindow, &(params.cannyThresh), 100, thresh_callback);
    //thresh_callback(0, &(params.cannyThresh));
    //ref_image.DisplayEnhanced(true);
    waitKey(0);

}

//1. Pyramid Down
//2. Find the eye and get the mask
//3. Histogram specification: 6535_left
//4. Histogram equalization (CLAHE) on V channel of the HSV image
//5. Resize to size x size
//6. Apply mask to filter background
//7. Write to out_path
void process_files(string& ref, fs::path& in_path, vector<string>& in_files, fs::path& out_path, Size& size)
{
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
        fs::path filePath = in_path / fs::path(in_file);
        // out-path
        fs::path outFilePath = out_path / fs::path(in_file);

        // read it
        rgb = imread(filePath.string(), IMREAD_COLOR);
        HaemoragingImage hi(rgb);
        // 1. Pyramid down
        hi.PyramidDown();
        hi.getEnhanced().copyTo(src);
        hi.setImage(src);

        // 2. Find contours, get mask.
        // if we did not get the entire eye - reset the threhsold to 1 and repeat
        int i = 0;
        do
        {
            hi.CreateEyeContours(i == 0 ? params.cannyThresh : 1);
            hi.CreateMask(size.width);
            i++;

        } while (hi.EyeAreaRatio() < 0.45 && i <= 2);

        // 3. Histogram specification
        Mat dest;
        histSpec.HistogramSpecification(src, dest);

        // 4. CLAHE
        hi.setImage(dest);
        hi.MakeHsv();
        hi.GetOneChannelImages(Channels::V);
        hi.ApplyClahe();
        
        src = hi.getEnhanced();

        // 5. resize
        resize(src, dest, size);
        cvtColor(dest, rgb, COLOR_HSV2BGR);

        // 6. apply mask
        hi.setImage(rgb);
        hi.MaskOffBackground();

        // 7. write out
        imwrite(outFilePath.string(), hi.getEnhanced());
    }
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    string in_dir = parser.get<string>("in");
    string out_dir = parser.get<string>("out");
    string ref = parser.get<string>("ref");
    params.cannyThresh = parser.get<int>("t");

    int dim = parser.get<int>("size");
    Size size(dim, dim);
    
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

    process_files(ref, in_path, in_files, out_path, size);
    return(0);
}

