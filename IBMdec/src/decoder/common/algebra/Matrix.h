#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <assert.h>

#include "MatrixBase.h"
#include "LogMessage.h"

namespace Bavieca {

/**
	@author daniel <dani.bolanos@gmail.com>
*/
template<class Real>
class Matrix : public MatrixBase<Real> {

	public:
		// allocate memory
		void allocate() {
			this->m_iStride = this->m_iCols;
			this->m_rData = new Real[this->getSize()];
		}
		
		// deallocate memory
		void deallocate() {
			assert(this->m_rData);
			delete [] this->m_rData;
			this->m_rData = NULL;
		}

	public:

		// constructor
		Matrix() : MatrixBase<Real>(0,0) {	
		}
		
		// constructor for square matrices
		Matrix(unsigned int iDim) : MatrixBase<Real>(iDim,iDim) {
			allocate();
			this->zero();
		}
		
		// constructor
		Matrix(unsigned int iRows, unsigned int iCols) : MatrixBase<Real>(iRows,iCols) {
			allocate();
			this->zero();
		}
		
		
		// constructor from another matrix
		Matrix(MatrixBase<Real> &m) : MatrixBase<Real>(m.getRows(),m.getCols()) {
			allocate();
			memcpy(this->m_rData,m.getData(),this->getSize()*sizeof(Real));
		}		
		
		// constructor from a symmetric matrix
		// destructor
		~Matrix() {
		
			if (MatrixBase<Real>::m_rData) {
				deallocate();
			}
		}
		
		
		// swap the contents of two matrices
		void swap(Matrix<Real> &m) {
		
			assert((this->m_iRows == m.getRows()) && (this->m_iCols == m.getCols()));
			std::swap(this->m_rData,m.m_rData);
		}
		
 
		
};

};	// end-of-namespace

#endif
