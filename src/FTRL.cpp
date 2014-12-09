#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <gflags/gflags.h>
#include "FTRL.h"

DEFINE_double(alpha, 0.01, "parameter alpha for ftrl optimization");
DEFINE_double(beta, 1, "parameter beta for ftrl optimization");
DEFINE_double(l1, 0.01, "l1 regularization");
DEFINE_double(l2, 0.01, "l2 regularization");

using namespace std;

inline int sgn(double val) {                                            
    return (0 < val) - (val < 0);                                             
} 
FTRL::FTRL(unsigned long d) {
    alpha = FLAGS_alpha;
    beta = FLAGS_beta;
    l1 = FLAGS_l1;
    l2 = FLAGS_l2;

    n = vector<double>(d,0);
    z = vector<double>(d,0);
    w = vector<double>(d,0);
}

double FTRL::PredictOne(const vector<int> &x) {
    double wTx = 0;
    for (auto& ind: x) {
        wTx += w[ind];
    }
    return 1.0 / (1.0 + exp(-max(min(wTx,35.0), -35.0)));
}

void FTRL::TrainOne(const vector<int> &x, const double y) {
    double wTx = 0;
    for (auto& ind: x) {
        double sign = sgn(z[ind]);
        // solve for w in closed form on a percoordinate bases:
        if (sign*z[ind] <= l1) {
            w[ind] = 0;
        } else {
            w[ind] = (sign*l1 - z[ind]) / ((beta + sqrt(n[ind])) / alpha + l2);
        }
        wTx += w[ind];
    }
    double p = 1.0 / (1.0 + exp(-max(min(wTx,35.0), -35.0)));
    double g = p-y; // gradient of loss w.r.t w
    double sigma;
    for (auto& ind: x) {
        sigma = (sqrt(n[ind] + g * g ) - sqrt(n[ind])) / alpha;
        z[ind] += g - sigma * w[ind];
        n[ind] += g * g;
    }
}

void FTRL::SaveModel(const string& model_file_name) {
    fstream model_stream;                                                      
    model_stream.open(model_file_name.c_str(), std::fstream::out);                        
    if (!model_stream) {                                                            
        cout<< "Error opening model output file " << model_file_name;                
        exit(1);                                                                      
    }                                                                               
    cout << "Writing model to: " << model_file_name;
    for (auto& x: w) {
        model_stream << x << std::endl;
    }
    model_stream.close();                                                           
}   

void FTRL::LoadModel(const string& model_file_name) {
    fstream model_stream;                                                      
    model_stream.open(model_file_name.c_str(), std::fstream::in);                        
    if (!model_stream) {                                                            
        cout<< "Error opening model output file " << model_file_name;                
        exit(1);                                                                      
    }                                                                 
    string line;
    while (getline(model_stream, line)) {
        w.emplace_back(stod(move(line)));
    }
    model_stream.close();
}




