//
// Created by Anurag Bhattacharyya on 10/23/24.
//

#ifndef GCMMA_H
#define GCMMA_H

#pragma once

#include <vector>
/**
 * GCMMASolver is a base class for performing all functions of GCMMA implementation
 */

class GCMMASolver {

public:
	GCMMASolver(int n,int m, double a = 0.0, double c = 1000.0, double d = 0.0);

	void SetAsymptotes(double init, double decrease, double increase);

	/**
	* Compute (L, U, raa0, raa), build and solve GCMMA subproblem
	* @param fx the approximation of the constraint
	* @param f0x the approximation of the objective function
	* @return void
	*/
	void OuterUpdate(double *xmma, const double *xval, double f0x, const double *df0dx,
		const double *fx, const double *dfdx, const double *xmin, const double *xmax);


	/**
	* Updates (L, U, raa0, raa), build and solve GCMMA subproblem
	* @param fx the approximation of the constraint
	* @param f0x the approximation of the objective function
	* @return void
	*/
	void InnerUpdate(double *xmma, double f0xnew, const double *fxnew,
		const double *xval, double f0x, const double *df0dx, const double *fx,
		const double *dfdx, const double *xmin, const double *xmax);

	/**
	* Checks whether the new solution is conservative or not
	*/
	bool ConCheck(double f0xnew, const double *fxnew) const;

	void Reset() { outeriter = 0; };

private:
	int n, m, outeriter;

	const double raa0eps;
	const double raaeps;
	const double xmamieps;
	const double epsimin;

	const double move, albefa;
	double asyminit, asymdec, asyminc;

	double raa0;
	std::vector<double> raa;

	std::vector<double> a, c, d;
	std::vector<double> y;
	double z;

	std::vector<double> lam, mu, s;
	std::vector<double> low, upp, alpha, beta, p0, q0, pij, qij, b, grad, hess;

	double r0, f0app;
	std::vector<double> r, fapp;

	std::vector<double> xold1, xold2;

private:
	/**
	* Compute
	*low, upp, raa0 and raa
	*/
	void Asymp(const double *xval, const double *df0dx,
		const double *dfdx, const double *xmin, const double *xmax);

	/**
	* Update
	* low, upp, raa0 and raa
	*/
	void RaaUpdate(const double *xmma, const double *xval, double f0xnew,
		const double *fxnew, const double *xmin, const double *xmax);

	/**
	* Build the GCMMA sub problem
	*/
	void GenSub(const double *xval, double f0x, const double *df0dx, const double *fx,
		const double *dfdx, const double *xmin, const double *xmax);

	/**
	* Compute approximations for objective and constraint functions
	*/
	void ComputeApprox(const double *xmma);

	void SolveDSA(double *x);
	void SolveDIP(double *x);

	void XYZofLAMBDA(double *x);

	void DualGrad(double *x);
	void DualHess(double *x);
	void DualLineSearch();
	double DualResidual(double *x, double epsi);

	static void Factorize(double *K, int n);
	static void Solve(double *K, double *x, int n);
};

#endif //GCMMA_H
