#include "stdafx.h"
#include "HaemoragingImage.hpp"

vector<vector<Point>> * FindHaemorages(unique_ptr<HaemoragingImage>& haemorage, Mat& image, ParamBag& params)
{
    haemorage->setImage(image);
    haemorage->GetGrayChannelImage();
    haemorage->GaussBlur();

    haemorage->DisplayEnhanced();

    //Find edges - first pass
    Mat edges;
    haemorage->GetCannyEdges(params.cannyThresh, params.cannyThresh * 3, edges);
    haemorage->FindBlobContours(edges);

    //Flatten what we've got
    auto contours = haemorage->FlattenContours();
    
    vector<vector<Point>> filtered(contours.size());

    auto it = std::copy_if(contours.begin(), contours.end(), filtered.begin(), [](vector<Point>& p){return p.size() > 3; });
    filtered.resize(distance(filtered.begin(), it));

    // try to close them off as much as possible
    vector<vector<Point>> polys(filtered.size());
    //close contours whenever possible
    std::transform(filtered.begin(), filtered.end(), polys.begin(), polys.begin(), [](vector<Point>& in, vector<Point>& out){ approxPolyDP(in, out, 3, true); return out; });
    haemorage->setContours(polys);

    //now flatten again 
    contours = haemorage->FlattenContours();
    it = std::copy_if(contours.begin(), contours.end(), filtered.begin(), [](vector<Point>& p){ auto area = contourArea(p); return area < 500; });
    filtered.resize(distance(filtered.begin(), it));

    vector<vector<Point>> * res = new vector<vector<Point>>(filtered.size());
    copy(filtered.begin(), filtered.end(), res->begin());

    return res;
}