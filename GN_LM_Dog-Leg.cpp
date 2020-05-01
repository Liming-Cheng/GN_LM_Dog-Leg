#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <iomanip>
#include <math.h>


using namespace std;
using namespace Eigen;

const double DERIV_STEP = 1e-5;
const int MAX_ITER = 100;

#define max(a,b) (((a)>(b))?(a):(b))

VectorXd objF(const VectorXd& input, const VectorXd& output, const VectorXd& params);
double func(const VectorXd& input, const VectorXd& output, const VectorXd& params, double objIndex);
MatrixXd Jacobin(const VectorXd& input, const VectorXd& output, const VectorXd& params);
double Deriv(const VectorXd& input, const VectorXd& output, int objIndex, const VectorXd& params,
                 int paraIndex);
double Func(const VectorXd& obj);
double maxMatrixDiagonale(const MatrixXd& Hessian);
double linerDeltaL(const VectorXd& step, const VectorXd& gradient, const double u);
void dogLeg(const VectorXd& input, const VectorXd& output, VectorXd& params);
void gaussNewton(const VectorXd& input, const VectorXd& output, VectorXd& params);
void levenMar(const VectorXd& input, const VectorXd& output, VectorXd& params);



int main(int argc, char* argv[])
{
    // obj = A * sin(Bx) + C * cos(D*x) - F
    //there are 4 parameter: A, B, C, D.
    int num_params = 4;

    //generate random data using these parameter
    int total_data = 100;

    VectorXd input(total_data);
    VectorXd output(total_data);

    double A = 5, B= 1, C = 10, D = 2;
    //load observation data
    for (int i = 0; i < total_data; i++)
    {
        //generate a random variable [-10 10]
        double x = 20.0 * ((random() % 1000) / 1000.0) - 10.0;
        double deltaY = 2.0 * (random() % 1000) /1000.0;
        double y = A*sin(B*x)+C*cos(D*x) + deltaY;

        input(i) = x;
        output(i) = y;
    }

    //gauss the parameters
    VectorXd params_gaussNewton(num_params);
    //init gauss
    params_gaussNewton << 1.6, 1.4, 6.2, 1.7;

    VectorXd params_levenMar = params_gaussNewton;
    VectorXd params_dogLeg   = params_gaussNewton;

    gaussNewton(input, output, params_gaussNewton);
    levenMar(input, output, params_levenMar);
    dogLeg(input, output, params_dogLeg);
    
    cout << endl;
    cout << "gauss newton parameter: " << endl << params_gaussNewton << endl << endl << endl << endl;
    cout << "Levenberg-Marquardt parameter: " << endl << params_levenMar << endl << endl << endl <<endl;
    cout << "dog-leg parameter: " << endl << params_dogLeg << endl << endl << endl <<endl;
}

///////////////////////////////////////////////////////////////////////////////// dog-leg //////////////////////////////////////////////////
void dogLeg(const VectorXd& input, const VectorXd& output, VectorXd& params)
{
    int errNum = input.rows();      //error num
    int paraNum = params.rows();    //parameter num

    VectorXd obj = objF(input, output, params);
    MatrixXd Jac = Jacobin(input, output, params);  //jacobin
    VectorXd gradient = Jac.transpose() * obj;      //gradient

    //initial parameter tao v epsilon1 epsilon2
    double eps1 = 1e-12, eps2 = 1e-12, eps3 = 1e-12;
    double radius = 1.0;

    bool found  = obj.norm() <= eps3 || gradient.norm() <= eps1;
    if(found) return;

    double last_sum = 0;
    int iterCnt = 0;
    while(iterCnt < MAX_ITER)
    {
        VectorXd obj = objF(input, output, params);
        MatrixXd Jac = Jacobin(input, output, params);  //jacobin
        VectorXd gradient = Jac.transpose() * obj;      //gradient

        if( gradient.norm() <= eps1 )
        {
            cout << "stop F'(x) = g(x) = 0 for a global minimizer optimizer." << endl;
            break;
        }
        if(obj.norm() <= eps3)
        {
            cout << "stop f(x) = 0 for f(x) is so small";
            break;
        }

        //compute how far go along stepest descent direction.
        double alpha = gradient.squaredNorm() / (Jac * gradient).squaredNorm();
        //compute gauss newton step and stepest descent step.
        VectorXd stepest_descent = -alpha * gradient;
        VectorXd gauss_newton = (Jac.transpose() * Jac).inverse() * Jac.transpose() * obj * (-1);

        double beta = 0;

        //compute dog-leg step.
        VectorXd dog_leg(params.rows());
        if(gauss_newton.norm() <= radius)
            dog_leg = gauss_newton;
        else if(alpha * stepest_descent.norm() >= radius)
            dog_leg = (radius / stepest_descent.norm()) * stepest_descent;
        else
        {
            VectorXd a = alpha * stepest_descent;
            VectorXd b = gauss_newton;
            double c = a.transpose() * (b - a);
            beta = (sqrt(c*c + (b-a).squaredNorm()*(radius*radius-a.squaredNorm()))-c)
                    /(b-a).squaredNorm();

            dog_leg = alpha * stepest_descent + beta * (gauss_newton - alpha * stepest_descent);

        }

        cout << "dog-leg: " << endl << dog_leg << endl;

        if(dog_leg.norm() <= eps2 *(params.norm() + eps2))
        {
            cout << "stop because change in x is small" << endl;
            break;
        }

        VectorXd new_params(params.rows());
        new_params = params + dog_leg;

        cout << "new parameter is: " << endl << new_params << endl;

        //compute f(x)
        obj = objF(input,output,params);
        //compute f(x_new)
        VectorXd obj_new = objF(input,output,new_params);

        //compute delta F = F(x) - F(x_new)
        double deltaF = Func(obj) - Func(obj_new);

        //compute delat L =L(0)-L(dog_leg)
        double deltaL = 0;
        if(gauss_newton.norm() <= radius)
            deltaL = Func(obj);
        else if(alpha * stepest_descent.norm() >= radius)
            deltaL = radius*(2*alpha*gradient.norm() - radius)/(2.0*alpha);
        else
        {
            VectorXd a = alpha * stepest_descent;
            VectorXd b = gauss_newton;
            double c = a.transpose() * (b - a);
            beta = (sqrt(c*c + (b-a).squaredNorm()*(radius*radius-a.squaredNorm()))-c)
                    /(b-a).squaredNorm();

            deltaL = alpha*(1-beta)*(1-beta)*gradient.squaredNorm()/2.0 + beta*(2.0-beta)*Func(obj);

        }

        double roi = deltaF / deltaL;
        if(roi > 0)
        {
            params = new_params;
        }
        if(roi > 0.75)
        {
            radius = max(radius, 3.0 * dog_leg.norm());
        }
        else if(roi < 0.25)
        {
            radius = radius / 2.0;
            if(radius <= eps2*(params.norm()+eps2))
            {
                cout << "trust region radius is too small." << endl;
                break;
            }
        }

        cout << "roi: " << roi << " dog-leg norm: " << dog_leg.norm() << endl;
        cout << "radius: " << radius << endl;

        iterCnt++;
        cout << "Iterator " << iterCnt << " times" << endl << endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////// lm /////////////////////////////////////////////////
double maxMatrixDiagonale(const MatrixXd& Hessian)
{
    int max = 0;
    for(int i = 0; i < Hessian.rows(); i++)
    {
        if(Hessian(i,i) > max)
            max = Hessian(i,i);
    }

    return max;
}

//L(h) = F(x) + h^t*J^t*f + h^t*J^t*J*h/2
//deltaL = h^t * (u * h - g)/2
double linerDeltaL(const VectorXd& step, const VectorXd& gradient, const double u)
{
    double L = step.transpose() * (u * step - gradient);
    return L/2;
}

void levenMar(const VectorXd& input, const VectorXd& output, VectorXd& params)
{
    int errNum = input.rows();      //error num
    int paraNum = params.rows();    //parameter num

    //initial parameter 
    VectorXd obj = objF(input,output,params);
    MatrixXd Jac = Jacobin(input, output, params);  //jacobin
    MatrixXd A = Jac.transpose() * Jac;             //Hessian
    VectorXd gradient = Jac.transpose() * obj;      //gradient

    //initial parameter tao v epsilon1 epsilon2
    double tao = 1e-3;
    long long v = 2;
    double eps1 = 1e-12, eps2 = 1e-12;
    double u = tao * maxMatrixDiagonale(A);
    bool found = gradient.norm() <= eps1;
    if(found) return;

    double last_sum = 0;
    int iterCnt = 0;

    while (iterCnt < MAX_ITER)
    {
        VectorXd obj = objF(input,output,params);

        MatrixXd Jac = Jacobin(input, output, params);  //jacobin
        MatrixXd A = Jac.transpose() * Jac;             //Hessian
        VectorXd gradient = Jac.transpose() * obj;      //gradient

        if( gradient.norm() <= eps1 )
        {
            cout << "stop g(x) = 0 for a local minimizer optimizer." << endl;
            break;
        }

        cout << "A: " << endl << A << endl; 

        VectorXd step = (A + u * MatrixXd::Identity(paraNum, paraNum)).inverse() * gradient; //negtive Hlm.

        cout << "step: " << endl << step << endl;

        if( step.norm() <= eps2*(params.norm() + eps2) )
        {
            cout << "stop because change in x is small" << endl;
            break;
        } 

        VectorXd paramsNew(params.rows());
        paramsNew = params - step; //h_lm = -step;

        //compute f(x)
        obj = objF(input,output,params);

        //compute f(x_new)
        VectorXd obj_new = objF(input,output,paramsNew);

        double deltaF = Func(obj) - Func(obj_new);
        double deltaL = linerDeltaL(-1 * step, gradient, u);

        double roi = deltaF / deltaL;
        cout << "roi is : " << roi << endl;
        if(roi > 0)
        {
            params = paramsNew;
            u *= max(1.0/3.0, 1-pow(2*roi-1, 3));
            v = 2;
        }
        else
        {
            u = u * v;
            v = v * 2;
        }

        cout << "u = " << u << " v = " << v << endl;

        iterCnt++;
        cout << "Iterator " << iterCnt << " times, result is :" << endl << endl;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////// gn /////////////////////////////////
double func(const VectorXd& input, const VectorXd& output, const VectorXd& params, double objIndex)
{
    // obj = A * sin(Bx) + C * cos(D*x) - F
    double x1 = params(0);
    double x2 = params(1);
    double x3 = params(2);
    double x4 = params(3);

    double t = input(objIndex);
    double f = output(objIndex);

    return x1 * sin(x2 * t) + x3 * cos( x4 * t) - f;
}

//return vector make up of func() element.
VectorXd objF(const VectorXd& input, const VectorXd& output, const VectorXd& params)
{
    VectorXd obj(input.rows());
    for(int i = 0; i < input.rows(); i++)
        obj(i) = func(input, output, params, i);

    return obj;
}

//F = (f ^t * f)/2
double Func(const VectorXd& obj)
{
    return obj.squaredNorm()/2;
}

double Deriv(const VectorXd& input, const VectorXd& output, int objIndex, const VectorXd& params,
                 int paraIndex)
{
    VectorXd para1 = params;
    VectorXd para2 = params;

    para1(paraIndex) -= DERIV_STEP;
    para2(paraIndex) += DERIV_STEP;

    double obj1 = func(input, output, para1, objIndex);
    double obj2 = func(input, output, para2, objIndex);

    return (obj2 - obj1) / (2 * DERIV_STEP);
}

MatrixXd Jacobin(const VectorXd& input, const VectorXd& output, const VectorXd& params)
{
    int rowNum = input.rows();
    int colNum = params.rows();

    MatrixXd Jac(rowNum, colNum);

    for (int i = 0; i < rowNum; i++)
    {
        for (int j = 0; j < colNum; j++)
        {
            Jac(i,j) = Deriv(input, output, i, params, j);
        }
    }
    return Jac;
}

void gaussNewton(const VectorXd& input, const VectorXd& output, VectorXd& params)
{
    int errNum = input.rows();      //error  num
    int paraNum = params.rows();    //parameter  num

    VectorXd obj(errNum);

    double last_sum = 0;

    int iterCnt = 0;
    while (iterCnt < MAX_ITER)
    {
        obj = objF(input, output, params);

        double sum = 0;
        sum = Func(obj);

        cout << "Iterator index: " << iterCnt << endl;
        cout << "parameter: " << endl << params << endl;
        cout << "error sum: " << endl << sum << endl << endl;

        if (fabs(sum - last_sum) <= 1e-12)
            break;
        last_sum = sum;

        MatrixXd Jac = Jacobin(input, output, params);
        VectorXd delta(paraNum);
        delta = (Jac.transpose() * Jac).inverse() * Jac.transpose() * obj;

        params -= delta;
        iterCnt++;
    }
}
