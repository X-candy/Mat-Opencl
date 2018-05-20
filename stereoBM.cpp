//M*//////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/****************************************************************************************\
*    Very fast SAD-based (Sum-of-Absolute-Diffrences) stereo correspondence algorithm.   *
*    Contributed by Kurt Konolige                                                        *
\****************************************************************************************/

#include "common.hpp"

//#include "precomp.hpp"
#include <stdio.h>
#include <limits>
//#include "opencl_kernels_calib3d.hpp"

#include "ocl_util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#if 0
	/*
	 * Please pay attention to modules/core/src/ocl.cpp file.
	 */
	int Kernel::set(int i, const KernelArg& arg)	// This function used to set up kernel arguments
	/*
	 * 	For WriteOnlyNoSize, We can use following method to calculate a UMat object's step and offset(Howard Chen 2018.03.19)
	 */
	UMat tmp_disp = disp.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
	tcout << "tmp_disp.step: " << tmp_disp.step << "tmp_disp.offset: " << tmp_disp.offset << endl;
#endif

#if 0
namespace cv
{
#endif

struct StereoBMParams
{
    StereoBMParams(int _numDisparities=64, int _SADWindowSize=21)
    {
        preFilterType 		= StereoBM::PREFILTER_XSOBEL;
        preFilterSize 		= 9;
        preFilterCap 		= 31;
        SADWindowSize 		= _SADWindowSize;
        minDisparity 		= 0;
        numDisparities 		= _numDisparities > 0 ? _numDisparities : 64;
        textureThreshold 	= 10;
        uniquenessRatio 	= 15;
        speckleRange 		= speckleWindowSize = 0;
        roi1 = roi2 		= Rect(0,0,0,0);
        disp12MaxDiff 		= -1;
        dispType 			= CV_16S;
    }

    int preFilterType;
    int preFilterSize;
    int preFilterCap;
    int SADWindowSize;
    int minDisparity;
    int numDisparities;
    int textureThreshold;
    int uniquenessRatio;
    int speckleRange;
    int speckleWindowSize;
    Rect roi1, roi2;
    int disp12MaxDiff;
    int dispType;

    // Howard 2018-03-08
    ocl::Program program;
    int			 m_sizeX;
};


static bool ocl_prefilter_norm(ocl::Program &program, InputArray _input, OutputArray _output, int winsize, int prefilterCap)
{
	ocl::Kernel k("prefilter_norm", program);
	if(k.empty())
		return false;

    int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
    scale_g *= scale_s;

    UMat input = _input.getUMat(), output;
    _output.create(input.size(), input.type());
    output = _output.getUMat();

    size_t globalThreads[3] = { (size_t)input.cols, (size_t)input.rows, 1 };

    k.args(ocl::KernelArg::PtrReadOnly(input), ocl::KernelArg::PtrWriteOnly(output), input.rows, input.cols,
        prefilterCap, scale_g, scale_s);

    return k.run(2, globalThreads, NULL, false);
}

static bool ocl_prefilter_xsobel(ocl::Program &program, InputArray _input, OutputArray _output, int prefilterCap)
{
	ocl::Kernel k("prefilter_xsobel", program);
	if(k.empty())
		return false;

    UMat input = _input.getUMat(), output;
    _output.create(input.size(), input.type());
    output = _output.getUMat();

    size_t globalThreads[3] = { (size_t)input.cols, (size_t)input.rows, 1 };

    k.args(ocl::KernelArg::PtrReadOnly(input), ocl::KernelArg::PtrWriteOnly(output), input.rows, input.cols, prefilterCap);

    return k.run(2, globalThreads, NULL, false);
}

static const int DISPARITY_SHIFT = 4;

static bool ocl_prefiltering(InputArray left0, InputArray right0, OutputArray left, OutputArray right, StereoBMParams* state)
{
    if( state->preFilterType == StereoBM::PREFILTER_NORMALIZED_RESPONSE )
    {
        if(!ocl_prefilter_norm( state->program, left0, left, state->preFilterSize, state->preFilterCap))
            return false;
        if(!ocl_prefilter_norm( state->program, right0, right, state->preFilterSize, state->preFilterCap))
            return false;
    }
    else
    {
        if(!ocl_prefilter_xsobel( state->program, left0, left, state->preFilterCap ))
            return false;
        if(!ocl_prefilter_xsobel( state->program, right0, right, state->preFilterCap))
            return false;
    }
    return true;
}


static bool ocl_stereobm(InputArray _left, InputArray _right,
                       OutputArray _disp, StereoBMParams* state)
{
    int ndisp = state->numDisparities;
    int mindisp = state->minDisparity;
    int wsz = state->SADWindowSize;
    int wsz2 = wsz/2;

	ocl::Kernel k("stereoBM", state->program);
	if(k.empty())
		return false;

	int sizeX = state->m_sizeX,
		sizeY = sizeX - 1,
		N = ndisp * 2;

    UMat left = _left.getUMat(), right = _right.getUMat();
    int cols = left.cols, rows = left.rows;

    _disp.create(_left.size(), CV_16S);
    _disp.setTo((mindisp - 1) << 4);
    Rect roi = Rect(Point(wsz2 + mindisp + ndisp - 1, wsz2), Point(cols-wsz2-mindisp, rows-wsz2) );
    UMat disp = (_disp.getUMat())(roi);

    //tcout << roi << endl;
    //tcout << "disp_cols: " << disp.cols << "	disp_rows: " << disp.rows <<endl;

    int globalX = (disp.cols + sizeX - 1) / sizeX,
        globalY = (disp.rows + sizeY - 1) / sizeY;
    size_t globalThreads[3] = {(size_t)N, (size_t)globalX, (size_t)globalY};
    size_t localThreads[3]  = {(size_t)N, 1, 1};

    //tcout << "N: " << N << "	globalX: " << globalX << "	globalY: " << globalY <<endl;

    int idx = 0;
    idx = k.set(idx, ocl::KernelArg::PtrReadOnly(left));
    idx = k.set(idx, ocl::KernelArg::PtrReadOnly(right));
    idx = k.set(idx, ocl::KernelArg::WriteOnlyNoSize(disp));
    idx = k.set(idx, rows);
    idx = k.set(idx, cols);
    idx = k.set(idx, state->textureThreshold);
    idx = k.set(idx, state->uniquenessRatio);
    return k.run(3, globalThreads, localThreads, false);
}


class StereoBMImpl : public StereoBM
{
public:
	StereoBMImpl()
    {
        params = StereoBMParams();
    }

	StereoBMImpl( int _numDisparities, int _SADWindowSize )
    {
        //tcout << "StereoBMImpl" << endl;
        params = StereoBMParams(_numDisparities, _SADWindowSize);

        //tcout << "Get context and build program" << endl;

        // Check if CL is active?
		if (!ocl::haveOpenCL())
		{
			tcout << "OpenCL is not avaiable..." << endl;
			return;
		}

		ocl::Context context;
		if (!context.create(ocl::Device::TYPE_ACCELERATOR))
		{
			tcout << "Failed creating the context..." << endl;
			return;
		}

		// Detect available devices.
        //tcout << context.ndevices() << " ACC devices are detected." << endl;
		for (int i = 0; i < context.ndevices(); i++)
		{
			cv::ocl::Device device = context.device(i);
            /*tcout << "name                 : " << device.name() << endl;
			tcout << "available            : " << device.available() << endl;
			tcout << "imageSupport         : " << device.imageSupport() << endl;
			tcout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
            tcout << endl;*/
		}

		// Select the first device
		ocl::Device(context.device(0));

		// Read the OpenCL kernel code
		ifstream ifs ("stereobm_hzhy.cl");
		if (ifs.fail()) return ;

		std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
		ocl::ProgramSource programSource(kernelSource);

		int ndisp = params.numDisparities;
		int wsz   = params.SADWindowSize;
		int sizeX = context.device(0).isIntel() ? 32 : std::max(11, 27 - context.device(0).maxComputeUnits()),
					sizeY = sizeX - 1,
					N = ndisp * 2;

		cv::String 		errmsg;
		cv::String 		opt = cv::format("-D DEFINE_KERNEL_STEREOBM -D MIN_DISP=%d -D NUM_DISP=%d"
										" -D BLOCK_SIZE_X=%d -D BLOCK_SIZE_Y=%d -D WSZ=%d",
										params.minDisparity, params.numDisparities,
										sizeX, sizeY, wsz);
		params.program = context.getProg(programSource, opt, errmsg);
		params.m_sizeX = sizeX;

        //tcout << "build option is :		" << opt.c_str() << endl << endl;
    }

    void compute( InputArray leftarr, InputArray rightarr, OutputArray disparr )
    {
        //tcout << "Enter StereoBMImpl' compute" << endl;

        int dtype = disparr.fixedType() ? disparr.type() : params.dispType;
        Size leftsize = leftarr.size();

        if (leftarr.size() != rightarr.size())
            CV_Error( Error::StsUnmatchedSizes, "All the images must have the same size" );

        if (leftarr.type() != CV_8UC1 || rightarr.type() != CV_8UC1)
            CV_Error( Error::StsUnsupportedFormat, "Both input images must have CV_8UC1" );

        if (dtype != CV_16SC1 && dtype != CV_32FC1)
            CV_Error( Error::StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format" );

        if( params.preFilterType != PREFILTER_NORMALIZED_RESPONSE &&
            params.preFilterType != PREFILTER_XSOBEL )
            CV_Error( Error::StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE" );

        if( params.preFilterSize < 5 || params.preFilterSize > 255 || params.preFilterSize % 2 == 0 )
            CV_Error( Error::StsOutOfRange, "preFilterSize must be odd and be within 5..255" );

        if( params.preFilterCap < 1 || params.preFilterCap > 63 )
            CV_Error( Error::StsOutOfRange, "preFilterCap must be within 1..63" );

        if( params.SADWindowSize < 5 || params.SADWindowSize > 255 || params.SADWindowSize % 2 == 0 ||
            params.SADWindowSize >= std::min(leftsize.width, leftsize.height) )
            CV_Error( Error::StsOutOfRange, "SADWindowSize must be odd, be within 5..255 and be not larger than image width or height" );

        if( params.numDisparities <= 0 || params.numDisparities % 16 != 0 )
            CV_Error( Error::StsOutOfRange, "numDisparities must be positive and divisble by 16" );

        if( params.textureThreshold < 0 )
            CV_Error( Error::StsOutOfRange, "texture threshold must be non-negative" );

        if( params.uniquenessRatio < 0 )
            CV_Error( Error::StsOutOfRange, "uniqueness ratio must be non-negative" );

       // tcout << "img_cols: " << leftarr.cols() << "	img_rows: " << leftarr.rows() << "	img_size: " << leftarr.size() << endl;

        int FILTERED = (params.minDisparity - 1) << DISPARITY_SHIFT;

        if(ocl::useOpenCL() && disparr.isUMat() && params.textureThreshold == 0)
        {
            //tcout << "Calling ocl_prefiltering" << endl;

            UMat left, right;
            if(ocl_prefiltering(leftarr, rightarr, left, right, &params))
            {
                //tcout << "Calling ocl_stereobm" << endl;
                if(ocl_stereobm(left, right, disparr, &params))
                {
                    //tcout << "disparr cols: " << disparr.cols() << "	rows: " << disparr.rows() << endl;

                    if( params.speckleRange >= 0 && params.speckleWindowSize > 0 )
                    {
                        //tcout << "Calling filterSpeckles" << endl;

                        filterSpeckles(disparr.getMat(), FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);
                    }
                    if (dtype == CV_32F)
                    {
                        //tcout << "Calling disparr.getUMat().convertTo" << endl;

                        disparr.getUMat().convertTo(disparr, CV_32FC1, 1./(1 << DISPARITY_SHIFT), 0);
                    }
                    CV_IMPL_ADD(CV_IMPL_OCL);
                }
            }
        }
        return;
    }

    int getMinDisparity() const { return params.minDisparity; }
    void setMinDisparity(int minDisparity) { params.minDisparity = minDisparity; }

    int getNumDisparities() const { return params.numDisparities; }
    void setNumDisparities(int numDisparities) { params.numDisparities = numDisparities; }

    int getBlockSize() const { return params.SADWindowSize; }
    void setBlockSize(int blockSize) { params.SADWindowSize = blockSize; }

    int getSpeckleWindowSize() const { return params.speckleWindowSize; }
    void setSpeckleWindowSize(int speckleWindowSize) { params.speckleWindowSize = speckleWindowSize; }

    int getSpeckleRange() const { return params.speckleRange; }
    void setSpeckleRange(int speckleRange) { params.speckleRange = speckleRange; }

    int getDisp12MaxDiff() const { return params.disp12MaxDiff; }
    void setDisp12MaxDiff(int disp12MaxDiff) { params.disp12MaxDiff = disp12MaxDiff; }

    int getPreFilterType() const { return params.preFilterType; }
    void setPreFilterType(int preFilterType) { params.preFilterType = preFilterType; }

    int getPreFilterSize() const { return params.preFilterSize; }
    void setPreFilterSize(int preFilterSize) { params.preFilterSize = preFilterSize; }

    int getPreFilterCap() const { return params.preFilterCap; }
    void setPreFilterCap(int preFilterCap) { params.preFilterCap = preFilterCap; }

    int getTextureThreshold() const { return params.textureThreshold; }
    void setTextureThreshold(int textureThreshold) { params.textureThreshold = textureThreshold; }

    int getUniquenessRatio() const { return params.uniquenessRatio; }
    void setUniquenessRatio(int uniquenessRatio) { params.uniquenessRatio = uniquenessRatio; }

    int getSmallerBlockSize() const { return 0; }
    void setSmallerBlockSize(int) {}

    Rect getROI1() const { return params.roi1; }
    void setROI1(Rect roi1) { params.roi1 = roi1; }

    Rect getROI2() const { return params.roi2; }
    void setROI2(Rect roi2) { params.roi2 = roi2; }

    void write(FileStorage& fs) const
    {
        fs << "name" << name_
        << "minDisparity" << params.minDisparity
        << "numDisparities" << params.numDisparities
        << "blockSize" << params.SADWindowSize
        << "speckleWindowSize" << params.speckleWindowSize
        << "speckleRange" << params.speckleRange
        << "disp12MaxDiff" << params.disp12MaxDiff
        << "preFilterType" << params.preFilterType
        << "preFilterSize" << params.preFilterSize
        << "preFilterCap" << params.preFilterCap
        << "textureThreshold" << params.textureThreshold
        << "uniquenessRatio" << params.uniquenessRatio;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert( n.isString() && String(n) == name_ );
        params.minDisparity = (int)fn["minDisparity"];
        params.numDisparities = (int)fn["numDisparities"];
        params.SADWindowSize = (int)fn["blockSize"];
        params.speckleWindowSize = (int)fn["speckleWindowSize"];
        params.speckleRange = (int)fn["speckleRange"];
        params.disp12MaxDiff = (int)fn["disp12MaxDiff"];
        params.preFilterType = (int)fn["preFilterType"];
        params.preFilterSize = (int)fn["preFilterSize"];
        params.preFilterCap = (int)fn["preFilterCap"];
        params.textureThreshold = (int)fn["textureThreshold"];
        params.uniquenessRatio = (int)fn["uniquenessRatio"];
        params.roi1 = params.roi2 = Rect();
    }

    StereoBMParams params;
    Mat preFilteredImg0, preFilteredImg1, cost, dispbuf;
    Mat slidingSumBuf;

    static const char* name_;
};

const char* StereoBMImpl::name_ = "StereoMatcher.BM";

Ptr<StereoBM> StereoBM::create(int _numDisparities, int _SADWindowSize)
{
    //tcout << "StereoBM::create" << endl;
    return makePtr<StereoBMImpl>(_numDisparities, _SADWindowSize);
}


void stereoBM_hzhy( InputArray _imgLeft,  InputArray _imgRight, OutputArray imgDisparity, \
				    int ndisparities, int SADWindowSize)
{
	Ptr<StereoBM> sbm = StereoBM::create( ndisparities, SADWindowSize );

	//-- 3. Calculate the disparity image
	sbm->setTextureThreshold(0);
	sbm->compute( _imgLeft, _imgRight, imgDisparity );
}

#if 0
}
#endif

/* End of file. */
