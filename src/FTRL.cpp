#include <cmath>
#include "FTRL.h"

using namespace std;

FTRL::FTRL(double alpha_, double beta_, double l1_, double l2_, unsigned long D_) {
    alpha = alpha_;
    beta = beta_;
    l1 = l1_;
    l2 = l2_;
    D = D_;

    n = vector<double>(D,0);
    z = vector<double>(D,0);
    w = vector<double>(D,0);
}

double FTRL::predict(vector<int> &x) {
    double wTx = 0;
    for (size_t i=0; i<x.size(); ++i) {
        int ind = x[i];
        double sign;
        if (z[ind]<0) {sign = -1;}
        else {sign = 1;}
        // solve for w in closed form on a percoordinate bases:
        if (sign*z[ind] <= l1) {
            w[ind] = 0;
        } else {
            w[ind] = (sign*l1 - z[ind]) / ((beta + sqrt(n[ind])) / alpha + l2);
        }
        wTx += w[ind];
    }

    return 1.0 / (1.0 + exp(-max(min(wTx,35.0), -35.0)));
}

void FTRL::update(vector<int> &x, double &p, double &y) {
    double g = p-y; // gradient of loss w.r.t w
    double sigma;
    for (size_t i=0; i<x.size(); i++) {
        int ind = x[i];
        sigma = (sqrt(n[ind] + g * g ) - sqrt(n[ind])) / alpha;
        z[ind] += g - sigma * w[ind];
        n[ind] += g * g;
    }
}






