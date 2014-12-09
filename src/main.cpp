#include <string>
#include <fstream>
#include <functional>
#include <sstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <ctime>
#include <iostream>
#include <gflags/gflags.h>
#include "FTRL.h"

DEFINE_string(test_file, "", "test file name");
DEFINE_string(training_file, "", "training file");
DEFINE_string(model_file, "", "model file");
DEFINE_string(submission_file, "", "submission  file");
DEFINE_bool(memory_limited, true, "is pc has enough memory");
DEFINE_uint64(D, 2<<20, "hash space");
DEFINE_int32(epochs, 4, "epoc num");
DEFINE_double(validation_Ration, 0.01, "validation ration");

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
    google::ParseCommandLineFlags(&argc, &argv, true);
    time_t t0 = time(nullptr);   // get time now
    

    double validationRatio = FLAGS_validation_Ration; // hold some training point for validation

    int epochs = 1;
    unsigned long D = FLAGS_D;
    FTRL learner(D);
    if (!FLAGS_model_file.empty()) {
        learner.LoadModel(FLAGS_model_file);
        D = learner.ModelSize();
    }
    hash<string> hashFun;
    if (!FLAGS_training_file.empty()) {
        double logloss_init = 100;
        if (FLAGS_memory_limited) {
            for (int epoch=0; epoch< FLAGS_epochs; epoch++) {
                // open the training file,
                ifstream file(FLAGS_training_file.c_str());
                string line;
                string cell;
                double loss = 0; // for validation
                int validationCounter = 0;
                int trainingCounter = 0;
                hash<string> hashFun;

                // read the data, one line each time
                dataPoint data;
                while(getline(file,line)) {
                    stringstream lineStream(line);
                    getline(lineStream, cell, ','); // first column is id
                    data.id = stoull(cell);

                    getline(lineStream, cell, ','); // 2nd column is click
                    if(cell=="1") data.y = 1;
                    else data.y = 0;
                    data.x.clear();
                    while(getline(lineStream,cell,',')) { // the remaining columns
                        data.x.push_back(hashFun(cell)%D);
                    }
                    if (validationRatio != 0) { // if we want have validation during training
                        if ((trainingCounter+1)%(int(1/validationRatio))==0) {
                            double p = learner.PredictOne(data.x);
                            loss += logLoss(p, data.y);
                            validationCounter += 1;
                        }
                        else {
                            learner.TrainOne(data.x, data.y);
                            trainingCounter += 1;
                        }
                    } else {
                        learner.TrainOne(data.x, data.y);
                        trainingCounter += 1;
                    }
                }
                file.close();
                if (validationRatio != 0) {
                    double ave_loss = loss/validationCounter;
                    cout<<"epoch: "<<epoch+1<<" finished, average loss: "<<loss/double(validationCounter)<<endl;
                    if (logloss_init - ave_loss < 0.00001) {
                        break;
                    } else {
                        logloss_init = ave_loss;
                    }
                }
                cout<<"time passed: "<< int(time(nullptr)) - int(t0)<<" sec"<<endl;
            }
        } else {
            ifstream file(FLAGS_training_file.c_str());
            string line;
            string cell;
            hash<string> hashFun;
            std::vector<dataPoint> dataSet;
            dataSet.reserve(1000000);
            // read the data, one line each time
            while(getline(file,line)) {
                dataPoint data;
                stringstream lineStream(line);
                getline(lineStream, cell, ','); // first column is id
                data.id = stoull(cell);

                getline(lineStream, cell, ','); // 2nd column is click
                if(cell=="1") data.y = 1;
                else data.y = 0;

                while(getline(lineStream,cell,',')) { // the remaining columns
                    data.x.push_back(hashFun(cell)%D);
                }
                dataSet.emplace_back(move(data));
            }
            for (int epoch=0; epoch< FLAGS_epochs; epoch++) {
                // open the training file,
                double loss = 0; // for validation
                int validationCounter = 0;
                int trainingCounter = 0;
                for (auto& data: dataSet) {
                    if (validationRatio != 0) { // if we want have validation during training
                        if ((trainingCounter+1)%(int(1/validationRatio))==0) {
                            double p = learner.PredictOne(data.x);
                            loss += logLoss(p, data.y);
                            validationCounter += 1;
                        }
                        else {
                            learner.TrainOne(data.x, data.y);
                            trainingCounter += 1;
                        }
                    } else {
                        learner.TrainOne(data.x, data.y);
                        trainingCounter += 1;
                    }
                }
                if (validationRatio != 0) {
                    double ave_loss = loss/validationCounter;
                    cout<<"epoch: "<<epoch+1<<" finished, average loss: "<<loss/double(validationCounter)<<endl;
                    if (logloss_init - ave_loss < 0.00001) {
                        break;
                    } else {
                        logloss_init = ave_loss;
                    }
                }
                cout<<"time passed: "<< int(time(nullptr)) - int(t0)<<" sec"<<endl;
            }
        }
        if (!FLAGS_model_file.empty()) {
            learner.SaveModel(FLAGS_model_file);
        }
    }
    if (!FLAGS_test_file.empty() && !FLAGS_submission_file.empty()) {
        ifstream test(FLAGS_test_file); // testing file
        ofstream submission(FLAGS_submission_file); // result file
        string line;
        string cell;
        // start predicting
        getline(test, line);
        submission<<"id,click"<<endl;
      
        dataPoint data;
        while(getline(test,line)) {
            stringstream lineStream(line);
            getline(lineStream, cell, ','); // 1st column is id
            data.id = stoull(cell);
            data.x.clear();
            while(getline(lineStream,cell,',')) { // remaining columns
                data.x.push_back(hashFun(cell)%D);
            }

            double p = learner.PredictOne(data.x);
            submission<<to_string(data.id)<<","<<to_string(p)<<endl;
        }
        test.close();
        submission.close();
    }
    return 0;
}






