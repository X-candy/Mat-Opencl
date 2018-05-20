#define __CL_ENABLE_EXCEPTIONS

#include "common.hpp"

#include "ocl_util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ti/cmem.h>
#include <sstream>

using namespace std;
using namespace cv;

// Define the macro for loop test times
#define 	LOOP_TEST_TIMES	4

extern void stereoBM(InputArray _imgLeft, InputArray _imgRight, OutputArray imgDisparity,
					 int ndisparities, int SADWindowSize);


/*
 * Time difference calculation, in ms units
 */
double tdiff_calc(struct timespec &tp_start, struct timespec &tp_end)
{
	return (double)(tp_end.tv_nsec -tp_start.tv_nsec) * 0.000001 + (double)(tp_end.tv_sec - tp_start.tv_sec) * 1000.0;
}

string int2str(int n)
{
    ostringstream ss;
    ss<<n;
    return ss.str();
}


/*
 * The app entry point
 */
int main(int argc, char** argv)
{
    // Initial  Umat 2 Mat
    struct timespec 	tp0, tp1, start, end;
    Mat					mat_raw, mat_rsz;
    double 				minVal, maxVal;
    AddrCvtReqHandle 	hAddrCvtReq;

    clock_gettime(CLOCK_MONOTONIC, &start);

    //  Read the images
    Mat imgLeft 	= imread( argv[1], IMREAD_GRAYSCALE );
    Mat imgRight 	= imread( argv[2], IMREAD_GRAYSCALE );

	cout << endl << "Begin UMat <-> Mat convert example (take OpenCv-OpenCl classes, this is to say option A)" << endl;
	cout << "imgLeft.cols: " << imgLeft.cols << "	imgLeft.rows: " << imgLeft.rows << endl;
	cout << "imgRight.cols: " << imgRight.cols << "	imgRight.rows: " << imgRight.rows << endl;

    //=======================SBM========================
    for(int zx=0; zx<LOOP_TEST_TIMES; zx++)
    {
        if(zx %2 ==0)
        {
            imgLeft 	= imread( argv[1], IMREAD_GRAYSCALE );
            imgRight 	= imread( argv[2], IMREAD_GRAYSCALE );
        }
        else
        {
            imgLeft 	= imread( argv[2], IMREAD_GRAYSCALE );
            imgRight 	= imread( argv[1], IMREAD_GRAYSCALE );
        }

		clock_gettime(CLOCK_MONOTONIC, &tp0);

		//  Allocate UMat
		UMat umat_Left 	= imgLeft.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
		UMat umat_Right = imgRight.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);

		//  And create the image in which we will save our disparities
		UMat umat_imgDisparity16S 	= UMat( imgLeft.rows, imgLeft.cols, CV_16S );
		Mat  mat_imgDisparity8U 	= Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );

		//-- 3. Call the constructor for StereoBM
		int ndisparities = 16*5;   	/**< Range of disparity */
		int SADWindowSize = 21; 	/**< Size of the block window. Must be odd */

		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("3. SBM initial  tdiff=%lf ms \n", tdiff_calc(tp0, tp1));

		//-- 4. Calculate the disparity image
		clock_gettime(CLOCK_MONOTONIC, &tp0);
		stereoBM_hzhy( umat_Left,  umat_Right, umat_imgDisparity16S, ndisparities, SADWindowSize);
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("4.StereoBM_OptionA  tdiff=%lf ms \n\n", tdiff_calc(tp0, tp1));

		printf("umat_imgDisparity16S.handle = 0x%x\n", umat_imgDisparity16S.handle(ACCESS_RW));

		string str1 = "SBM_umat_imgDisparity16S_loop_"+int2str(zx)+".png";
		imwrite(str1.c_str(), umat_imgDisparity16S);

		//-- Check its extreme values
		clock_gettime(CLOCK_MONOTONIC, &tp0);
		minMaxLoc( umat_imgDisparity16S, &minVal, &maxVal );
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("5. minMaxLoc  tdiff=%lf ms;  Min disp: %f Max value: %f\n", tdiff_calc(tp0, tp1), minVal, maxVal);

		clock_gettime(CLOCK_MONOTONIC, &tp0);
		umat_imgDisparity16S.convertTo( mat_imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf("mat_imgDisparity8U.ptr()=0x%x\n", mat_imgDisparity8U.ptr());
		printf ("6. Umat 16S convertTo Mat 8U tdiff=%lf ms \n\n", tdiff_calc(tp0, tp1));

		string str2 = "SBM_mat_imgDisparity8U_loop_"+int2str(zx)+".png";
		imwrite(str2.c_str(), mat_imgDisparity8U);

		/*
		 * UMat convert to Mat
		 */
		double gt_maxValue = maxVal, gt_minVal=minVal;
		clock_gettime(CLOCK_MONOTONIC, &tp0);
		Mat mat_imgDisparity16S = umat_imgDisparity16S.getMat(ACCESS_RW);
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("7. HS_UMat2Mat(UMat16S->Mat16S)  tdiff=%lf ms \n", tdiff_calc(tp0, tp1));

		string str4 = "SBM_mat_imgDisparity16S_loop_"+int2str(zx)+".png";
		imwrite(str4.c_str(), mat_imgDisparity16S);

		clock_gettime(CLOCK_MONOTONIC, &tp0);
		minMaxLoc( mat_imgDisparity16S, &minVal, &maxVal );
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("8. minMaxLoc  tdiff=%lf ms; Min disp: %f Max value: %f \n", tdiff_calc(tp0, tp1), minVal, maxVal);

		// Release mat
		mat_imgDisparity16S.release();

		clock_gettime(CLOCK_MONOTONIC, &tp0);
		UMat umat_imgDisparity8U;
		umat_imgDisparity16S.convertTo(umat_imgDisparity8U, CV_8UC1, 255/(maxVal- minVal));
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("9. Umat 16S convertTo Umat 8U  tdiff=%lf ms \n", tdiff_calc(tp0, tp1));

		string str5 = "SBM_umat_imgDisparity8U_loop_"+int2str(zx)+".png";
		imwrite(str5.c_str(), umat_imgDisparity8U);

		/*
		 * UMat convert to Mat
		 */
		clock_gettime(CLOCK_MONOTONIC, &tp0);
		Mat mat_cvt_imgDisparity8U = umat_imgDisparity8U.getMat(ACCESS_RW);
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		printf ("10. HS_UMat2Mat(UMat8U->Mat8U)  tdiff=%lf ms \n", tdiff_calc(tp0, tp1));

		string str6 = "SBM_mat_cvt_imgDisparity8U_loop_"+int2str(zx)+".png";
		imwrite(str6.c_str(), mat_cvt_imgDisparity8U);

		// Release mat
		mat_cvt_imgDisparity8U.release();

		clock_gettime(CLOCK_MONOTONIC, &end);
		printf ("end:  Sum Time   tdiff=%lf ms \n\n", tdiff_calc(start, end));
    }

	cout << "Exit UMat <-> Mat convert example" << endl << endl;
    return 1;
}
