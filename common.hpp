/*
 * hzhyapp.hpp
 *
 *  Created on: 2018-3-6
 *      Author: Howard
 */

#ifndef HZHYAPP_HPP_
#define HZHYAPP_HPP_

#include <CL/cl.hpp>

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <cassert>
#include <time.h>
#include <unistd.h>
#include <functional>


#define __TRACE__


#ifdef __TRACE__
#define tcout cout
#else
#define tcout 0 && cout
#endif


typedef enum {
    UMAT2UMAT,
    UMAT2MAT,
    MAT2UMAT,
    MAT2MAT
} CovertType;

typedef struct _tagVirAddrInfor{
	void * ptr_vir;
	void * ptr_phy;
	int	   bufsize;
}VirAddrInforObj, *VirAddrInforHandle;

typedef struct _tagAddrCvtReq{
#define	MAXBUFNUM 2
	VirAddrInforObj 	VirAddrInfor[MAXBUFNUM];
	int			   		numReq;		// must <= MAXBUFNUM
}AddrCvtReqObj, *AddrCvtReqHandle;

#endif /* HZHYAPP_HPP_ */
