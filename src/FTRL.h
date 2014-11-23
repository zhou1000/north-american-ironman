#include <vector>
#include <map>
#include <iostream>

// Yiqian, 20141122
//  FTRL algorithm: Follow the regularized leader - proximal
//  Reference:
//  http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf

class FTRL{
    public:
        double alpha;
        double beta;
        double l1;
        double l2;
        unsigned long D;

        std::vector<double> n;
        std::vector<double> z;
        std::vector<double> w;
        FTRL(double alpha_ = 0.1, double beta_ = 1, double l1_ = 1, double l2_ = 1, unsigned long D_ = 2<<10);
        double predict(std::vector<int> &x);
        void update(std::vector<int> &x, double &p, double &y);
};



