#pragma once
#include "TransformImage.hpp"

class HaemoragingImage : public TransformImage
{
protected:
    Scalar color_white;
    Mat _mask;
    vector<Point> _hull;

    void MergeContourPoints(vector<vector<Point>>& contour, vector<Point>& merged)
    {
        for (int i = 0; i < contour.size(); i++)
        {
            for (int j = 0; j < contour[i].size(); j++)
            {
                merged.push_back(contour[i][j]);
            }
        }
    }

public:
 
    HaemoragingImage(Mat& image) : TransformImage(image), color_white(255, 255, 255) {}
    HaemoragingImage() : TransformImage(), color_white(255, 255, 255) {}

    vector<vector<Point>>& FlattenContours()
    {
        Mat black = Mat::zeros(_image.rows, _image.cols, CV_8UC1);
        DrawContours(_contours, _hierarchy, black, 2);
        _contours.clear();
        _hierarchy.clear();

        return FindBlobContours(black);
    }

    vector<Point>& CreateEyeContours(int thresh)
    {
        //(gpu) grey channel image messes with the stored channel
        Channels channel = _channel;
        // grey channel
        GetGrayChannelImage();

        //(gpu) blur
        GaussBlur();

        //(gpu) canny edges
        Mat edges;
        GetCannyEdges(thresh, thresh * 3, edges);

        //contours
        auto contours = FindBlobContours(edges);

        // merge all contours into one vector
        vector<Point> merged_contour_points;
        MergeContourPoints(contours, merged_contour_points);

        // get rotated bounding box
        convexHull(Mat(merged_contour_points), _hull);

        _channel = channel;
        return _hull;
    }

    Mat& DrawEyeContours(Mat& image, Scalar& color = Scalar(0, 0, 255), int thickness = CV_FILLED)
    {
        if (image.rows == 0 || _hull.size() == 0)
        {
            throw logic_error("image not initialized or hull not computed");
        }

        vector<vector<Point>> hull(1);
        hull[0] = _hull;

        TransformImage::DrawContours(hull, vector<Vec4i>(), image, thickness, color);
        return image;
    }

    Mat& CreateMask(int size = -1)
    {
        if (_hull.size() == 0)
        {
            throw logic_error("Hull has not been computed yet");
        }

        Size maskSize(_image.cols, _image.rows);
        _mask = Mat::zeros(maskSize, CV_8UC1);
        DrawEyeContours(_mask, color_white);

        if (size > 0)
        {
            Size newSize(size, size);
            resize(_mask, _mask, newSize);
        }
        
        return _mask;
    }

    double EyeAreaRatio()
    {
        if (_mask.size() == Size(0, 0))
        {
            throw logic_error("Need to compute the eye mask first");
        }

        double mean;

        gpu::GpuMat gMask(_mask), gBuf;

        mean = gpu::countNonZero(gMask, gBuf);
        return mean / (_mask.rows * _mask.cols);
    }

    void MaskOffBackground()
    {
        if (_mask.rows == 0)
        {
            throw logic_error("Need to compute mask first");
        }

        gpu::GpuMat g_mask(_mask);
        g_image.copyTo(g_enhanced, g_mask);
    }
};