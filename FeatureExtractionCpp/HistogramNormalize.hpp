#pragma once
#include "TransformImage.hpp"

// histogram specification & equalization for color images
class HistogramNormalize
{

private:
    TransformImage _refImage;
    gpu::GpuMat g_hist;
    const Channels channels[3] = { Channels::H, Channels::S, Channels::V };

public:
    HistogramNormalize(Mat refImage) : _refImage(refImage)  { }

    void CalcRefHistograms()
    {
        _refImage.MakeHsv();

        // we are working only with the intensity channel
        _refImage.GetOneChannelImages(Channels::V);

        _refImage.GetHist(g_hist);
    }



};