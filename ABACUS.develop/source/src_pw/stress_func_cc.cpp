#include "./stress_func.h"
#include "./H_XC_pw.h"
#include "../src_global/math_integral.h"

//NLCC term, need to be tested
void Stress_Func::stress_cc(
	matrix& sigma,
	const bool is_pw,
	PW_Basis &pwb)
{
	timer::tick("Stress_Func","stress_cc",'F');
        
	double fact=1.0;

	if(is_pw&&INPUT.gamma_only) 
	{
		fact = 2.0; //is_pw:PW basis, gamma_only need to double.
	}

	complex<double> sigmadiag;
	double* rhocg;
	double g[3];

	int judge=0;
	for(int nt=0;nt<ucell.ntype;nt++)
	{
		if(ucell.atoms[nt].nlcc) 
		{
			judge++;
		}
	}

	if(judge==0) 
	{
		return;
	}

	//recalculate the exchange-correlation potential
    const auto etxc_vtxc_v = H_XC_pw::v_xc(pwb.nrxx, pwb.ncxyz, ucell.omega, CHR.rho, CHR.rho_core);
	H_XC_pw::etxc    = std::get<0>(etxc_vtxc_v);			// may delete?
	H_XC_pw::vtxc    = std::get<1>(etxc_vtxc_v);			// may delete?
	const matrix vxc = std::get<2>(etxc_vtxc_v);

	complex<double> * psic = new complex<double> [pwb.nrxx];

	ZEROS(psic, pwb.nrxx);

	if(NSPIN==1||NSPIN==4)
	{
		for(int ir=0;ir<pwb.nrxx;ir++)
		{
			// psic[ir] = vxc(0,ir);
			psic[ir] = complex<double>(vxc(0, ir),  0.0);
		}
	}
	else
	{
		for(int ir=0;ir<pwb.nrxx;ir++)
		{
			psic[ir] = 0.5 * (vxc(0, ir) + vxc(1, ir));
		}
	}

	// to G space
	pwb.FFT_chg.FFT3D(psic, -1);

	//psic cantains now Vxc(G)
	rhocg= new double [pwb.nggm];
	ZEROS(rhocg, pwb.nggm);

	sigmadiag=0.0;
	for(int nt=0;nt<ucell.ntype;nt++)
	{
		if(ucell.atoms[nt].nlcc)
		{
			//drhoc();
			CHR.non_linear_core_correction(
				ppcell.numeric,
				ucell.atoms[nt].msh,
				ucell.atoms[nt].r,
				ucell.atoms[nt].rab,
				ucell.atoms[nt].rho_atc,
				rhocg);


			//diagonal term 
			if (pwb.gstart==1) 
			{
				sigmadiag += conj(psic [pwb.ig2fftc[0]] ) * pwb.strucFac (nt, 0) * rhocg [pwb.ig2ngg[0] ];
			}
			for(int  ng = pwb.gstart;ng< pwb.ngmc;ng++)
			{
				sigmadiag +=  conj(psic[pwb.ig2fftc[ng]] ) *
					pwb.strucFac (nt, ng) * rhocg [pwb.ig2ngg[ng] ] * fact;
			}
			this->deriv_drhoc (
				ppcell.numeric,
				ucell.atoms[nt].msh,
				ucell.atoms[nt].r,
				ucell.atoms[nt].rab,
				ucell.atoms[nt].rho_atc,
				rhocg,
				pw);
			// non diagonal term (g=0 contribution missing)
			for(int  ng = pwb.gstart;ng< pwb.ngmc;ng++)
			{
				g[0] = pwb.gcar[ng].x;
				g[1] = pwb.gcar[ng].y;
				g[2] = pwb.gcar[ng].z;
				for(int  l = 0;l< 3;l++)
				{
					for (int m = 0;m< 3;m++)
					{
						const complex<double> t = conj(psic[pwb.ig2fftc[ng]] )
							* pwb.strucFac (nt, ng) * rhocg [pwb.ig2ngg[ng] ] * ucell.tpiba *
							g [l] * g [m] / pwb.gcar[ng].norm() * fact;
//						sigmacc [l][ m] += t.real();
						sigma(l,m) += t.real();
					}//end m
				}//end l
			}//end ng
		}//end if
	}//end nt

	for(int  l = 0;l< 3;l++)
	{
		sigma(l,l) += sigmadiag.real();
//		sigmacc [l][ l] += sigmadiag.real();
	}
	for(int  l = 0;l< 3;l++)
	{
		for (int m = 0;m< 3;m++)
		{
			Parallel_Reduce::reduce_double_pool( sigma(l,m) );
		}
	}

	delete[] rhocg;
	delete[] psic;

	timer::tick("Stress_Func","stress_cc");
	return;
}


void Stress_Func::deriv_drhoc 
(
	const bool &numeric,
	const int mesh,
	const double *r,
	const double *rab,
	const double *rhoc,
	double *drhocg,
	PW_Basis &pwb
)
{

	double gx = 0;
	double rhocg1 = 0;
	// the modulus of g for a given shell
	// the fourier transform
	double *aux = new double[ mesh];
	// auxiliary memory for integration

	int  igl0 = 0;
	// counter on radial mesh points
	// counter on g shells
	// lower limit for loop on ngl

	//
	// G=0 term
	//
	if (pwb.ggs[0] < 1.0e-8)
	{
		drhocg [0] = 0.0;
		igl0 = 1;
	}
	else
	{
		igl0 = 0;
	}
	//
	// G <> 0 term
	//
	
	for(int igl = igl0;igl< pwb.nggm;igl++)
	{
		gx = sqrt(pwb.ggs [igl] * ucell.tpiba2);
		for( int ir = 0;ir< mesh; ir++)
		{
			aux [ir] = r [ir] * rhoc [ir] * (r [ir] * cos (gx * r [ir] ) / gx - sin (gx * r [ir] ) / pow(gx,2));
		}//ir
		Integral::Simpson_Integral(mesh, aux, rab, rhocg1);
		drhocg [igl] = FOUR_PI / ucell.omega * rhocg1;
	}//igl
	
	delete [] aux;

	return;
}
