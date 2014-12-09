#include <iostream>

#include "DataReader.h"
#include "DataWriter.h"
#include "GBDT.h"

using namespace std;

int main() {
    DataReader datareader;
    DataWriter datawriter;

    vector<vector<double>> Xtrain = datareader.readCSVdata("train_subsample_0001.csv");
    vector<double> target(Xtrain.size());

    for (size_t i=0; i<Xtrain.size(); ++i) {
        target[i] = Xtrain[i][1];
        Xtrain[i].erase(Xtrain[i].begin()+1);
    }
    
    vector<vector<double>> Xtest = datareader.readCSVdata("test_subsample_001.csv");
    for (size_t i=0; i<Xtest.size(); ++i) {
        Xtest[i].erase(Xtest[i].begin());
    }

    GBDT learner;
    
    learner.trainGBDT(Xtrain,target);

    vector<vector<double>> res(Xtest.size(),vector<double>(2));
    for (size_t i=0; i<Xtest.size(); ++i) {
        res[i][0] = learner.predictGBDT(Xtest[i]);
    }
    vector<string> header = {"results"};
    datawriter.writeCSV("tmp_result.csv", header , res);
    return 0;
}


