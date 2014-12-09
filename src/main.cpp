#include <string>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>
#include <limits>
#include <cmath>
#include <ctime>
#include <iostream>
#include "FTRL.h"


using namespace std;

struct dataPoint {
    unsigned long long id;
    double y;
    vector<int> x;
};

double logLoss(double p, double y) {
    p = max(min(p, 1 - numeric_limits<double>::min()), numeric_limits<double>::min());
    // min() 2.22507e-308
    // lowest() -1.79769e+3081.45243
    if (y==1) {
        return -log(p);
    } else {
        return -log(1-p);
    }
}

int main(int argc, char *argv[]) {
    time_t t0 = time(nullptr);   // get time now
    
    string usage_str ="usage: main traintrain.csv testfile.csv submissionfile.csv validationRatio";
    if (argc!=5) {cout<<usage_str<<endl; return 1;}

    ifstream test(argv[2]); // testing file
    ofstream submission(argv[3]); // result file
    double validationRatio = stod(argv[4]); // hold some training point for validation

    int epochs = 1;
    for (int epoch=0; epoch< epochs; epoch++) {
        // parameters for FTRL training
        double alpha = 0.1;
        double beta = 1;
        double l1 = 1;
        double l2 = 1;
        unsigned long D = 1<<20;

        FTRL learner(alpha, beta, l1, l2, D);

        // open the training file,
        ifstream file(argv[1]);
        string line;
        string cell;

        double loss = 0; // for validation
        int validationCounter = 0;
        int trainingCounter = 0;
        hash<string> hashFun;

        // read the data, one line each time
        getline(file,line); //skip the header
        while(getline(file,line)) {
            dataPoint data;
            stringstream lineStream(line);
            getline(lineStream, cell, ','); // first column is id
            data.id = stoull(cell);

            getline(lineStream, cell, ','); // 2nd column is click
            if(cell=="1") data.y = 1;
            else data.y = 0;

            getline(lineStream, cell, ','); // 3rd column is hour, but we don't care YYMMDD, only use HH
            data.x.push_back(hashFun(cell.substr(6,2))%D); // hash trick

            while(getline(lineStream,cell,',')) { // the remaining columns
                data.x.push_back(hashFun(cell)%D);
            }
            double p = learner.predict(data.x);
            if (validationRatio != 0) { // if we want have validation during training
                if ((trainingCounter+1)%(int(1/validationRatio))==0) {
                    loss += logLoss(p, data.y);
                    validationCounter += 1;
                }
                else {
                    learner.update(data.x, p, data.y);
                    trainingCounter += 1;
                }
            } else {
                learner.update(data.x, p, data.y);
                trainingCounter += 1;
            }
        }
        if (validationRatio != 0) {
            cout<<"epoch: "<<epoch+1<<" finished, average loss: "<<loss/double(validationCounter)<<endl;
        }

        // start predicting

        getline(test, line);
        submission<<"id,click"<<endl;

        while(getline(test,line)) {
            dataPoint data;
            stringstream lineStream(line);
            getline(lineStream, cell, ','); // 1st column is id
            data.id = stoull(cell);

            getline(lineStream, cell, ','); // 2nd column is YYMMDDHH
            data.x.push_back(hashFun(cell.substr(6,2))%D); // we don't care YYMMDD, only use HH

            while(getline(lineStream,cell,',')) { // remaining columns
                data.x.push_back(hashFun(cell)%D);
            }

            double p = learner.predict(data.x);
            submission<<to_string(data.id)<<","<<to_string(p)<<endl;
        }

        // done one epoch
        cout<<"time passed: "<< int(time(nullptr)) - int(t0)<<" sec"<<endl;
    }
    return 0;
}























