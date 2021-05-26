#include "./stress_pw.h"
#include "./H_XC_pw.h"
#include "src_pw/vdwd2.h"
#include "src_pw/vdwd3.h"

void Stress_PW::cal_stress(matrix& sigmatot)
{
	TITLE("Stress_PW","cal_stress");
	timer::tick("Stress_PW","cal_stress",'E');    

	// total stress
	sigmatot.create(3,3);
	matrix sigmaxc;
	// exchange-correlation stress
	sigmaxc.create(3,3);
	// hartree stress
	matrix sigmahar;
	sigmahar.create(3,3);
	// electron kinetic stress
	matrix sigmakin;
	sigmakin.create(3,3);
	// local pseudopotential stress
	matrix sigmaloc;
	sigmaloc.create(3,3);
	// non-local pseudopotential stress
	matrix sigmanl;
	sigmanl.create(3,3);
	// Ewald stress
	matrix sigmaewa;
	sigmaewa.create(3,3);
	// non-linear core correction stress
	matrix sigmaxcc;
	sigmaxcc.create(3,3);
	// vdw stress
	matrix sigmavdw;
	sigmavdw.create(3,3);

	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			sigmatot(i,j) = 0.0;
			sigmaxc(i,j) = 0.0;
			sigmahar(i,j) = 0.0;
			sigmakin(i,j) = 0.0;
			sigmaloc(i,j) = 0.0;
			sigmanl(i,j) = 0.0;
			sigmaewa(i,j) = 0.0;
			sigmaxcc(i,j) = 0.0;
			sigmavdw(i,j) = 0.0;
		}
	}

	//kinetic contribution
	stress_kin(sigmakin, kv);
	
	//hartree contribution
	stress_har(sigmahar, 1);

    //ewald contribution
    stress_ewa(sigmaewa, 1);

    //xc contribution: add gradient corrections(non diagonal)
    for(int i=0;i<3;i++)
	{
       sigmaxc(i,i) = - (H_XC_pw::etxc - H_XC_pw::vtxc) / ucell.omega;
    }
    stress_gga(sigmaxc);

    //local contribution
    stress_loc(sigmaloc, 1);
    
    //nlcc
    stress_cc(sigmaxcc, 1);
   
    //nonlocal
	stress_nl(sigmanl, NBANDS, wf.npwx);

	//vdw term
	stress_vdw(sigmavdw);

    for(int ipol=0;ipol<3;ipol++)
	{
        for(int jpol=0;jpol<3;jpol++)
		{
			sigmatot(ipol,jpol) = sigmakin(ipol,jpol) 
								+ sigmahar(ipol,jpol) 
								+ sigmanl(ipol,jpol) 
								+ sigmaxc(ipol,jpol) 
								+ sigmaxcc(ipol,jpol) 
								+ sigmaewa(ipol,jpol)
								+ sigmaloc(ipol,jpol);
								+ sigmavdw(ipol,jpol);
        }
    }
    
	if(Symmetry::symm_flag)                          
	{
		symm.stress_symmetry(sigmatot);
	}

	bool ry = false;
	this->printstress_total(sigmatot, ry);

	if(TEST_STRESS) 
	{               
		ofs_running << "\n PARTS OF STRESS: " << endl;
		ofs_running << setiosflags(ios::showpos);
		ofs_running << setiosflags(ios::fixed) << setprecision(10) << endl;
		this->print_stress("KINETIC    STRESS",sigmakin,TEST_STRESS,ry);
		this->print_stress("LOCAL    STRESS",sigmaloc,TEST_STRESS,ry);
		this->print_stress("HARTREE    STRESS",sigmahar,TEST_STRESS,ry);
		this->print_stress("NON-LOCAL    STRESS",sigmanl,TEST_STRESS,ry);
		this->print_stress("XC    STRESS",sigmaxc,TEST_STRESS,ry);
		this->print_stress("EWALD    STRESS",sigmaewa,TEST_STRESS,ry);
		this->print_stress("NLCC    STRESS",sigmaxcc,TEST_STRESS,ry);
		this->print_stress("TOTAL    STRESS",sigmatot,TEST_STRESS,ry);
	}
	return;
    
}

void Stress_PW::stress_vdw(matrix& sigma)
{
	matrix force;
	if(vdwd2_para.flag_vdwd2) //Peize Lin add 2014-04-04, update 2021-03-09
	{
		Vdwd2 vdwd2(ucell,vdwd2_para);
		vdwd2.cal_stress();
		sigma = vdwd2.get_stress().to_matrix();
	}
	if(vdwd3_para.flag_vdwd3) //jiyy add 2019-05-18, update 2021-05-02
	{
		Vdwd3 vdwd3(ucell,vdwd3_para);
		vdwd3.cal_stress();
		sigma = vdwd3.get_stress().to_matrix();
	}              
	return;
}
