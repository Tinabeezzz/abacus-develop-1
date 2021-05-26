#include "./stress_func.h"
#include "./myfunc.h"
#include "./H_Hartree_pw.h"

//calculate the Hartree part in PW or LCAO base
void Stress_Func::stress_har(
	matrix& sigma, 
	const bool is_pw,
	PW_Basis &pwb)
{
	timer::tick("Stress_Func","stress_har",'F');
	double shart,g2;
	const double eps=1e-8;
	int l,m,nspin0;

	complex<double> *Porter = UFFT.porter;

	//  Hartree potential VH(r) from n(r)
	ZEROS( Porter, pwb.nrxx );
	for(int is=0; is<NSPIN; is++)
	{
		for (int ir=0; ir<pwb.nrxx; ir++)
		{
			Porter[ir] += complex<double>( CHR.rho[is][ir], 0.0 );
		}
	}
	//=============================
	//  bring rho (aux) to G space
	//=============================
	pwb.FFT_chg.FFT3D(Porter, -1);

	complex<double> *psic = new complex<double> [pwb.nrxx];
	double *psic0 = new double[pwb.nrxx];
	ZEROS( psic0, pwb.nrxx);
	for(int is=0; is<NSPIN; is++)
	{
		daxpy (pwb.nrxx, 1.0, CHR.rho[is], 1, psic0, 2);
		for (int ir=0; ir<pwb.nrxx; ir++)
		{
			psic[ir] = complex<double>(psic0[ir], 0.0);
		}
	}

	pwb.FFT_chg.FFT3D(psic, -1) ;


	double charge;
	if (pwb.gstart == 1)
	{
		charge = ucell.omega * Porter[pwb.ig2fftc[0]].real();
	}

	complex<double> *vh_g  = new complex<double>[pwb.ngmc];
	ZEROS(vh_g, pwb.ngmc);

	double g[3];

	for (int ig = pwb.gstart; ig<pwb.ngmc; ig++)
	{
		const int j = pwb.ig2fftc[ig];
		const double fac = e2 * FOUR_PI / (ucell.tpiba2 * pwb.gg [ig]);

		shart= ( conj( Porter[j] ) * Porter[j] ).real()/(ucell.tpiba2 * pwb.gg [ig]);
		g[0]=pwb.gcar[ig].x;
		g[1]=pwb.gcar[ig].y;
		g[2]=pwb.gcar[ig].z;

		for(int l=0;l<3;l++)
		{
			for(int m=0;m<l+1;m++)
			{
				sigma(l,m) += shart *2*g[l]*g[m]/pwb.gg[ig];
			}
		}
	}

//	Parallel_Reduce::reduce_double_pool( en.ehart );
//	ehart *= 0.5 * ucell.omega;
        //cout<<"ehart "<<ehart<<" en.ehart "<< en.ehart<<endl;
	for(int l=0;l<3;l++)
	{
		for(int m=0;m<l+1;m++)
		{
			Parallel_Reduce::reduce_double_pool( sigma(l,m) );
		}
	}

//        Parallel_Reduce::reduce_double_pool( ehart );
//        ehart *= 0.5 * ucell.omega;
        //psic(:)=(0.0,0.0)

	if(is_pw && INPUT.gamma_only)
	{
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<3;m++)
			{
				sigma(l,m) *= e2 * FOUR_PI;
			}
		}
	}
	else
	{
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<3;m++)
			{
				sigma(l,m) *= 0.5 * e2 * FOUR_PI;
			}
		}
	}
	
	for(int l=0;l<3;l++)
	{
		if(is_pw) 
		{
			sigma(l,l) -= H_Hartree_pw::hartree_energy /ucell.omega;
		}
		else 
		{
			sigma(l,l) += H_Hartree_pw::hartree_energy /ucell.omega;
		}
		for(int m=0;m<l;m++)
		{
			sigma(m,l)=sigma(l,m);
		}
	}
	
	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			sigma(l,m)*=-1;
		}
	}

	delete[] vh_g;
	delete[] psic;
	delete[] psic0;
	timer::tick("Stress_Func","stress_har");
	return;
}
