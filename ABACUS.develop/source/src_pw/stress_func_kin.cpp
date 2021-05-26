#include"stress_func.h"

//calculate the kinetic stress in PW base
void Stress_Func::stress_kin(
	matrix& sigma,
	kvect &kp)
{
	double *kfac;
	double **gk;
	gk=new double*[3];

	double tbsp=2.0/sqrt(PI);
	double gk2=0.0;
	double arg=0.0;

	double s_kin[3][3];

	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			s_kin[l][m]=0.0;
		}
	}
		
	int npwx=0;
	int qtot = 0;
	for(int ik=0; ik<kp.nks; ik++)
	{
		for(int ig=0; ig<kp.ngk[ik]; ig++)
		{
			qtot += kp.ngk[ik];
		}
		if(npwx<kp.ngk[ik])
		{
			npwx=kp.ngk[ik];
		}
	}
		
	kfac=new double[npwx];
	gk[0]= new double[npwx]; 
	gk[1]= new double[npwx];
	gk[2]= new double[npwx];

	for(int i=0;i<npwx;i++)
	{
		kfac[i]=1;
	}

	double factor=TWO_PI/ucell.lat0;

	for(int ik=0;ik<kp.nks;ik++)
	{
		// number of plane waves per k-point
		int npw = kp.ngk[ik];

		for(int i=0;i<npw;i++)
		{
			gk[0][i]=(kp.kvec_c[ik].x+pw.gcar[wf.igk(ik, i)].x)*factor;
			gk[1][i]=(kp.kvec_c[ik].y+pw.gcar[wf.igk(ik, i)].y)*factor;
			gk[2][i]=(kp.kvec_c[ik].z+pw.gcar[wf.igk(ik, i)].z)*factor;
			  
	//          if(qcutz>0){
	//             gk2=pow(gk[i].x,2)+pow(gk[i].y,2)+pow(gk[i].z,2);
	//             arg=pow((gk2-ecfixed)/q2sigma,2);
	//             kfac[i]=1+qcutz/q2sigma*tbsp*exp(-arg);
	//          }
		}

		//kinetic contribution
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<l+1;m++)
			{
				for(int ibnd=0;ibnd<NBANDS;ibnd++)
				{
					for(int i=0;i<npw;i++)
					{
						if(0)
						{
							s_kin[l][m] +=
								wf.wg(ik,ibnd)*gk[l][i]*gk[m][i]*kfac[i]
								*(double((conj(wf.evc[ik](ibnd, i))
								*wf.evc[ik](ibnd, i)).real())+
								double((conj(wf.evc[ik](ibnd, i))*wf.evc[ik](ibnd, i+npwx)).real()));
						}
						else
						{
							s_kin[l][m] +=
								wf.wg(ik, ibnd)*gk[l][i]*gk[m][i]*kfac[i]
								*(double((conj(wf.evc[ik](ibnd, i))*wf.evc[ik](ibnd, i)).real()));
						}
					}
				}
			}
		}
		   
		//contribution from the nonlocal part
		//stres_us(ik, gk, npw);
	}
		
	//add the US term from augmentation charge derivatives
		
	// addussstres(sigmanlc);
	
	//mp_cast
		
	for(int l=0;l<3;l++)
	{
		for(int m=0;m<l;m++)
		{
			s_kin[m][l]=s_kin[l][m];
		}
	}

	if(INPUT.gamma_only)
	{
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<3;m++)
			{
				s_kin[l][m] *= 2.0*e2/ucell.omega;
			}
		}
	}
	else 
	{
		for(int l=0;l<3;l++)
		{
			for(int m=0;m<3;m++)
			{
				s_kin[l][m] *= e2/ucell.omega;
			}
		}
	}

	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			Parallel_Reduce::reduce_double_pool( s_kin[l][m] );
		}
	}

	for(int l=0;l<3;l++)
	{
		for(int m=0;m<3;m++)
		{
			sigma(l,m) = s_kin[l][m];
		}
	}

	//do symmetry
	if(Symmetry::symm_flag)
	{
		symm.stress_symmetry(sigma);
	}//end symmetry
	
	delete[] kfac;
	delete[] gk[0];
	delete[] gk[1];
	delete[] gk[2];
	delete[] gk;
		
	return;
}
