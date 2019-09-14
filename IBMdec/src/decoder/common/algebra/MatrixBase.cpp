#include "MatrixBase.h"
#include "IOBase.h"
#include "Matrix.h"
#include "VectorStatic.h"

#include <stdlib.h>

namespace Bavieca {

// compute derivative (iDelta = size of the regression window)
template<typename Real> void MatrixBase<Real>::delta(MatrixBase<Real> &mDelta, unsigned int iDelta) {
	// precompute the constant
	Real rDiv = 0.0;
	for(unsigned int i=1 ; i <= iDelta ; ++i) {
		rDiv += i*i;
	}
	// compute the derivatives
	for(unsigned int i = 0 ; i < getRows() ; ++i) {
		VectorStatic<Real> vDelta = mDelta.getRow(i);
		vDelta.zero();
		for(unsigned int j = 1 ; j <= iDelta ; ++j) {
			vDelta.add(j,getRow(std::min(i+j,getRows()-1)));
			vDelta.add(-((int)j),getRow(std::max(0,(int)(i-j))));
		}
		vDelta.mul(1.0/rDiv);
	}	
}

// return whether all the elements in the matrix are actual finite numbers  
template<>bool MatrixBase<float>::finite() {
	for(unsigned int i=0; i < m_iRows; ++i) {
		for(unsigned int j=0; j < m_iCols ; ++j) {
			if (finitef((*this)(i,j)) == 0) {
				return false;
			}
		}
	}
	return true;
}

// set to zero
template<typename Real>
void MatrixBase<Real>::zero() {
	for(unsigned int i=0; i < m_iRows; ++i) {
		for(unsigned int j=0; j < m_iCols ; ++j) {
			m_rData[i*m_iStride+j] = 0.0;
		}
	}
}

template void MatrixBase<float>::zero();
template void MatrixBase<double>::zero();


// return whether all the elements in the matrix are actual finite numbers  
//template<>bool MatrixBase<double>::finite() { return true;}


template void MatrixBase<float>::delta(MatrixBase<float> &mDelta, unsigned int iDelta);
 

};	// end-of-namespace


