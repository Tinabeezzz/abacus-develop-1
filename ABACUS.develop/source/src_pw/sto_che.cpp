#include "sto_che.h" 
#include "diago_cg.h"
#include "global.h"
#include "../src_global/constants.h"

Stochastic_Chebychev::Stochastic_Chebychev()
{
    initplan = false;
    initcoef = false;
    getcoef = false;
    getpolyval = false;
    extend = 16;
    norder = 5;
    polyvalue = new double [1];
    coef = new double [1];
    ccoef = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) *1);
    dcoef = (double *) fftw_malloc(sizeof(double) * 1);
}

Stochastic_Chebychev::~Stochastic_Chebychev()
{
    if(initplan)
    {
        fftw_destroy_plan(plancoef);
	}
	fftw_free(ccoef);
	fftw_free(dcoef);
	delete [] polyvalue;
	delete [] coef;
}

void Stochastic_Chebychev::init(int &dim, int &chetype)
{
    this->ndim = dim;
    norder = STO_WF.nche_sto;

    if(norder<5)
    {
        WARNING_QUIT("Stochastic_Chebychev", "The Chebychev expansion order should be at least 5!");
    }

    assert(extend >= 1);
    if(chetype==2)
    {
        vecn = new complex<double> [norder * dim];
        ZEROS(vecn,norder * dim);
    }
    fftw_free(ccoef);
    fftw_free(dcoef);
    delete [] polyvalue;
    delete [] coef;
    norder2 = 2 * norder * extend;
    ccoef = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * norder2);
    dcoef = (double *) fftw_malloc(sizeof(double) * norder2);
    plancoef = fftw_plan_dft_r2c_1d(norder2, dcoef, ccoef, FFTW_ESTIMATE);

	assert(norder>0); //mohan add 2021-05-08
    polyvalue = new double [norder];
    coef = new double [norder];
    initcoef = true;
    initplan = true;
    return;
}
    
void Stochastic_Chebychev::calcoef(double fun(double))
{
    if(!initcoef) WARNING_QUIT("Stochastic_Chebychev", "Please init coef first!");
    complex<double> *pcoef = (complex<double> *)ccoef;
    //three point = 2/3 M + 1/3 T;

    //(M)iddle point integral method part
    for(int i = 0; i < norder2; ++i)
    {
        dcoef[i]=fun(cos((i+0.5)*TWO_PI/norder2));
    }

    fftw_execute(plancoef);
    complex<double> ui(0,1.0);
    for(int i = 0; i<norder; ++i)
    {
        if(i == 0)
        {
            coef[i] = real(exp(-ui*i*PI/norder2) * pcoef[i]) / norder2 * 2 / 3;
        }
        else
        {
            coef[i] = real(exp(-ui*i*PI/norder2) * pcoef[i]) / norder2 * 4 / 3;
        }
    }

    //(T)rapezoid integral method part
    for(int i = 0; i < norder2; ++i)
    {
        dcoef[i]=fun(cos(i*TWO_PI/norder2));
    }
    
    fftw_execute(plancoef);
    for(int i = 0; i<norder; ++i)
    {
        if(i == 0)
        {
            coef[i] += real(pcoef[i]) / norder2 * 1 / 3;
        }
        else
        {
            coef[i] += real(pcoef[i]) / norder2 * 2 / 3;
        }
    }
    if( coef[norder-1]/coef[0] > 1e-9 )
    {
        WARNING("Stochastic_Chebychev", 
		"(coef[norder-1]/coef[0] > 1e-9) Please add more expansion terms for Chebychev expansion.");
    }
    getcoef = true;
	return;
}


complex<double> Stochastic_Chebychev::sumallterms(void)
{
    if(!getcoef||!getpolyval) 
	{
		WARNING_QUIT("Stochastic_Chebychev", "Please calculate coef or polyval first!");
	}

    complex<double> result = 0;
    for(int ior = 0; ior < norder; ++ior)
    {
        result += coef[ior] * polyvalue[ior];
    }
    return result;   
}


void Stochastic_Chebychev::calpolyval(
	void tfun(complex<double> *in, complex<double> *out, const int), 
	complex<double> *wavein, 
	const int m)
{
    complex<double> *arraynp1;
	complex<double> *arrayn;
	complex<double> *arrayn_1;

    int ndmx = ndim * m;

    arraynp1 = new complex<double> [ndmx];
    arrayn = new complex<double> [ndmx];
    arrayn_1 = new complex<double> [ndmx];

    DCOPY(wavein, arrayn_1, ndmx);

    tfun(arrayn_1, arrayn,m);

    polyvalue[0] = Diago_CG::ddot_real(ndmx,wavein,wavein, false);
    polyvalue[1] = Diago_CG::ddot_real(ndmx,wavein,arrayn, false);
    
    //ZEROS(polyvalue,norder);
    ////0- & 1-st order
    //for(int i = 0; i < ndim; ++i) 
    //{
    //    polyvalue[0] += real(conj(wavein[i]) * wavein[i]);// 0-th order : <wavein | wavein> = ndim
    //    polyvalue[1] += real(conj(wavein[i]) * arrayn[i]);// 1-st order : <wavein | tfun | wavein>
    //}


    //more than 1-st orders
    for(int ior = 2; ior < norder; ++ior)
    {
        recurs(arraynp1, arrayn, arrayn_1, tfun, m);
        polyvalue[ior] = Diago_CG::ddot_real(ndmx,wavein,arraynp1, false);
        
        //for(int i = 0; i < ndim; ++i) // n-th order : <wavein | T_n(tfun) | wavein>
        //{
        //    polyvalue[ior] += real(conj(wavein[i]) * arraynp1[i]);
        //}
        
        complex<double>* tem = arrayn_1;
        arrayn_1 = arrayn;
        arrayn = arraynp1;
        arraynp1 = tem; 
    }

    delete [] arraynp1;
    delete [] arrayn;
    delete [] arrayn_1;
    getpolyval = true;
    return;
}

void Stochastic_Chebychev::calfinalvec(
	void tfun(complex<double> *in, complex<double> *out, const int n), 
	complex<double> *wavein, 
	complex<double> *waveout, 
	const int m)
{
    if(!getcoef) WARNING_QUIT("Stochastic_Chebychev", "Please calculate coef first!");

    complex<double> *arraynp1;
	complex<double> *arrayn;
	complex<double> *arrayn_1;

    int ndmx = ndim * m;

    arraynp1 = new complex<double> [ndmx];
    arrayn = new complex<double> [ndmx];
    arrayn_1 = new complex<double> [ndmx];
  
    DCOPY(wavein, arrayn_1, ndmx);
    
    tfun(arrayn_1, arrayn,m);
    
    //0- & 1-st order
    for(int i = 0; i < ndmx; ++i)
    {
        waveout[i] = coef[0] * arrayn_1[i] + coef[1] * arrayn[i];
    }    

    //more than 1-st orders
    for(int ior = 2; ior < norder; ++ior)
    {
        recurs(arraynp1, arrayn, arrayn_1, tfun, m);
        for(int i = 0; i < ndmx; ++i)
        {
            waveout[i] += coef[ior] * arraynp1[i];
        }
        complex<double> * tem = arrayn_1;
        arrayn_1 = arrayn;
        arrayn = arraynp1;
        arraynp1 = tem; 
    }
    delete [] arraynp1;
    delete [] arrayn;
    delete [] arrayn_1;
    return;
}


bool Stochastic_Chebychev::checkconverge(
	void tfun(complex<double> *in, complex<double> *out, const int), 
	complex<double> *wavein,
	double& tmax, 
	double &tmin, 
	double stept)
{
    bool converge = true;
    complex<double> *arraynp1;
	complex<double> *arrayn;
	complex<double> *arrayn_1;

    arraynp1 = new complex<double> [ndim];
    arrayn = new complex<double> [ndim];
    arrayn_1 = new complex<double> [ndim];

    DCOPY(wavein, arrayn_1, ndim);
    //LapackConnector::copy(ndim,wavein,1,arrayn_1,1); 
    if(tmin == tmax) 
	{
		tmax += stept;
	}

    tfun(arrayn_1, arrayn,1);
    double sum1,sum2;
    double t;

    sum1=Diago_CG::ddot_real(ndim,arrayn_1,arrayn_1);
    sum2=Diago_CG::ddot_real(ndim,arrayn_1,arrayn);
    t = sum2 / sum1 * (tmax - tmin) / 2 + (tmax + tmin) / 2;
    if(t < tmin)
    {
        converge = false;
        tmin = t-stept;
    }
    else if(t > tmax)
    {
        converge = false;
        tmax = t+stept;
    }

    for(int ior = 2; ior < norder; ++ior)
    {
        tfun(arrayn,arraynp1,1);
        sum1=Diago_CG::ddot_real(ndim,arrayn,arrayn);
        sum2=Diago_CG::ddot_real(ndim,arrayn,arraynp1);
        t = sum2/sum1 * (tmax - tmin) / 2 + (tmax + tmin) / 2;
        if(t < tmin)
        {
            converge = false;
            tmin = t - stept;
        }
        else if(t > tmax)
        {
            converge = false;
            tmax = t + stept;
        }
        for(int i = 0; i < ndim; ++i)
        {
            arraynp1[i]=2*arraynp1[i]-arrayn_1[i];
        }
        complex<double>* tem = arrayn_1;
        arrayn_1 = arrayn;
        arrayn = arraynp1;
        arraynp1 = tem; 
    }
  
    delete [] arraynp1;
    delete [] arrayn;
    delete [] arrayn_1;
    return converge;
}
