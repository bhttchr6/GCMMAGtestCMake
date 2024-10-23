#include "gtest/gtest.h"
#include "../src/gcmma.h"

double Squared(double x) { return x*x; }

struct Problem {
    int n, m;
    std::vector<double> x0, xmin, xmax;

    Problem()
        : n(3)
        , m(2)
        , x0({4, 3, 2})
        , xmin(n, 0.0)
        , xmax(n, 5.0)
    { }

    void Obj(const double *x, double *f0x, double *fx) {
        f0x[0] = 0;
        for (int i = 0; i < n; ++i) {
            f0x[0] += x[i]*x[i];
        }
        fx[0] = Squared(x[0] - 5) + Squared(x[1] - 2) + Squared(x[2] - 1) - 9;
        fx[1] = Squared(x[0] - 3) + Squared(x[1] - 4) + Squared(x[2] - 3) - 9;
    }

    void ObjSens(const double *x, double *f0x, double *fx, double *df0dx, double *dfdx) {
        Obj(x, f0x, fx);
        for (int i = 0; i < n; ++i) {
            df0dx[i] = 2*x[i];
        }
        int k = 0;
        dfdx[k++] = 2 * (x[0] - 5); dfdx[k++] = 2 * (x[0] - 3);
        dfdx[k++] = 2 * (x[1] - 2); dfdx[k++] = 2 * (x[1] - 4);
        dfdx[k++] = 2 * (x[2] - 1); dfdx[k++] = 2 * (x[2] - 3);
    }
};

double VarChange() {
    Problem toy;
    double movlim = 0.2;

    double f, fnew;
    std::vector<double> df(toy.n);
    std::vector<double> g(toy.m), gnew(toy.m);
    std::vector<double> dg(toy.n * toy.m);

    std::vector<double> x = toy.x0;
    std::vector<double> xold = x;
    std::vector<double> xnew(toy.n);

    GCMMASolver gcmma(toy.n, toy.m, 0, 1000, 1);

    double ch = 1.0;
    int maxoutit = 8;
    for (int iter = 0; ch > 0.0002 && iter < maxoutit; ++iter) {
        toy.ObjSens(x.data(), &f, g.data(), df.data(), dg.data());

        // Call the update method

        // GCMMA version
        gcmma.OuterUpdate(xnew.data(), x.data(), f, df.data(),
            g.data(), dg.data(), toy.xmin.data(), toy.xmax.data());

        // Check conservativity
        toy.Obj(xnew.data(), &fnew, gnew.data());
        bool conserv = gcmma.ConCheck(fnew, gnew.data());
        //std::cout << conserv << std::endl;
        for (int inneriter = 0; !conserv && inneriter < 15; ++inneriter) {
            // Inner iteration update
            gcmma.InnerUpdate(xnew.data(), fnew, gnew.data(), x.data(), f,
                df.data(), g.data(), dg.data(), toy.xmin.data(), toy.xmax.data());

            // Check conservativity
            toy.Obj(xnew.data(), &fnew, gnew.data());
            conserv = gcmma.ConCheck(fnew, gnew.data());
            //std::cout << conserv << std::endl;
        }
        x = xnew;


        // Compute infnorm on design change
        ch = 0.0;
        for (int i=0; i < toy.n; ++i) {
            ch = std::max(ch, std::abs(x[i] - xold[i]));
            xold[i] = x[i];
        }

    }
    return ch;
}
// Demonstrate some basic assertions.
TEST(TestArithmetic, testadd) {
    // Expect equality.
    double expected = VarChange();
    double ground_truth = 1e-03;
    // The value returned from the GCMMA should be less than the ground_truth
    EXPECT_LT(expected,ground_truth);
}

