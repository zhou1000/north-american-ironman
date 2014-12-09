#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <numeric>

#include "GBDT.h"

using namespace std;

bool compareNodeReduced (nodeReduced n0, nodeReduced n1 ) {
    return n0.size < n1.size;
}

bool comparePair( const pair<double, double> &l, const pair<double, double> &r) {
    return l.first < r.first;
}

GBDT::GBDT() {
    featureSubspaceSize = 15;
    maxTreeLeaves = 1000;
    lRate = 0.1;
    doOptSplit = false;
    nEpochs = 1000;
}

GBDT::~GBDT() {
    cout<<"deconstructing..."<<endl;
}


double GBDT::predictSingleTree(node* n, vector<double> oneSample) {
    int ind = n->featureInd;
    // todo: add checkpoint, -1 <= ind <= nFeatures

    if (n->smallChild == nullptr && n->largeChild == nullptr) {
        return n->value; // at leaf node
    }

    double cutoff = n->value;
    double val = oneSample[ind];
    if (val <= cutoff) {
        return predictSingleTree(n->smallChild, oneSample);
    }
    return predictSingleTree(n->largeChild, oneSample);
}


void GBDT::trainSingleTree(node* n, deque<nodeReduced> &nodesToSplit
                           , vector<vector<double>> const &X
                           , vector<double> const &target
                           , vector<bool> usedFeatures) {

    unsigned int nNodeSamples = n->trainSamples.size();
    unsigned int nFeatures = X[0].size();

    // check it's ok to split the current node
    if (nodesToSplit.size() >= maxTreeLeaves || nNodeSamples<=1)
        return;

    // remove the current node from the heap
    if (nodesToSplit.size() > 0) {
        pop_heap(nodesToSplit.begin(), nodesToSplit.end(), compareNodeReduced);
        nodesToSplit.pop_back();
    }

    // select features randomly
    vector<int> selectedFeatures(featureSubspaceSize);

    if (featureSubspaceSize < nFeatures) {
        for (unsigned int j=0; j<featureSubspaceSize; ++j) {
            unsigned int ind = rand() % nFeatures;
            while (usedFeatures[ind])
                ind = rand() % nFeatures;
            selectedFeatures[j] = ind;
            usedFeatures[ind] = true;
        }
    } else { // featureSubspaceSize is larger than available, so use them all
        for (unsigned int j=0; j<featureSubspaceSize; ++j)
            selectedFeatures[j] = j;
    }

    // sums and squared sums of targets. before
    double sumTarget = 0.0, sum2Target = 0.0;
    for (unsigned int i=0; i<nNodeSamples; ++i) {
        double v = target[n->trainSamples[i]];
        sumTarget += v;
        sum2Target += v*v;
    }

    // now ready to select feature for this node
    int bestFeature = -1;
    //bestFeaturePos = -1;
    //    double bestFeatureLow = 1e10, bestFeatureHi = 1e10; // not used
    double bestFeatureRMSE = 1e10;
    double optFeatureSplitValue = 1e10;

    // search optimal split for each feature
    vector<int> selectedSamples = n->trainSamples;
    for (unsigned j=0; j<featureSubspaceSize; ++j) {
        int featureInd = selectedFeatures[j];

        double optimalSplitValue = 0.0;
        double rmseBest = 1e10;
        // double meanLowBest = 1e10, meanHiBest = 1e10;
        int bestPos = -1;
        double sumLow = 0.0, sum2Low = 0.0;
        double sumHi = sumTarget, sum2Hi = sum2Target;
        double cntLow = 0.0, cntHi = nNodeSamples;

        if (doOptSplit == false) {
            bestPos = rand() % nNodeSamples; // yiqian: randomly select the split position
            optimalSplitValue = X[selectedSamples[bestPos]][featureInd];
            sumLow = 0.0;
            sum2Low = 0.0;
            cntLow = 0.0;
            sumHi = 0.0;
            sum2Hi = 0.0;
            cntHi = 0.0;
            for ( unsigned int i=0; i<nNodeSamples; i++ )
            {
                double t = target[selectedSamples[i]];
                if ( t <= optimalSplitValue ) {
                    sumLow += t;
                    sum2Low += t*t;
                    cntLow += 1.0;
                }
                else {
                    sumHi += t;
                    sum2Hi += t*t;
                    cntHi += 1.0;
                }
            }
            rmseBest = ( sum2Low/cntLow - ( sumLow/cntLow ) * ( sumLow/cntLow ) ) *cntLow; // yiqian: !!! var(X) = E(X^2) - E(X)^2
            rmseBest += ( sum2Hi/cntHi - ( sumHi/cntHi ) * ( sumHi/cntHi ) ) *cntHi;
            rmseBest = sqrt ( rmseBest/ ( cntLow+cntHi ) );
            // meanLowBest = sumLow/cntLow;
            // meanHiBest = sumHi/cntHi;
        } else {
            vector<pair<double,double>> Xy(nNodeSamples); // a tmpX used for sorting
            for (unsigned i=0; i<nNodeSamples; ++i) {
                Xy[i].first = X[selectedSamples[i]][featureInd];
                Xy[i].second = target[selectedSamples[i]];
            }
            sort(Xy.begin(), Xy.end(), comparePair);

            unsigned int i = 0;
            while ( i < nNodeSamples-1 ) // yiqian: go through each possible split point
            {
                double t = Xy[i].second;
                sumLow += t;
                sum2Low += t*t;
                sumHi -= t;
                sum2Hi -= t*t;
                cntLow += 1.0;
                cntHi -= 1.0;

                double v0 = Xy[i].first, v1 = 1e10;
                if ( i < nNodeSamples -1 )
                    v1 = Xy[i+1].first;
                if ( v0 == v1 ) // skip equal successors
                {
                    i++;
                    continue;
                }

                double rmse = ( sum2Low/cntLow - ( sumLow/cntLow ) * ( sumLow/cntLow ) ) *cntLow;
                rmse += ( sum2Hi/cntHi   - ( sumHi/cntHi ) * ( sumHi/cntHi ) ) *cntHi;
                rmse = sqrt ( rmse/ ( cntLow+cntHi ) );

                if ( rmse < rmseBest )
                {
                    optimalSplitValue = v0;
                    rmseBest = rmse;
                    bestPos = i+1;
                    // meanLowBest = sumLow/cntLow;
                    // meanHiBest = sumHi/cntHi;
                }
                j++;
            }
        } // now get the best split for this feature

        if ( rmseBest < bestFeatureRMSE ) // yiqian: record the best split point of best feature
        {
            bestFeature = j;
            // bestFeaturePos = bestPos;
            bestFeatureRMSE = rmseBest;
            optFeatureSplitValue = optimalSplitValue;
            // bestFeatureLow = meanLowBest;
            // bestFeatureHi = meanHiBest;
        }
    }

    // unmark the selected
    for ( unsigned int j=0; j<nFeatures; j++ )
        usedFeatures[j] = false;

    // update the current node
    n->featureInd = selectedFeatures[bestFeature];
    n->value = optFeatureSplitValue;

    if ( n->featureInd < 0 || n->featureInd >= nFeatures ) {
        cout<<"f="<<n->featureInd<<endl;
        assert ( false );
    }

    // count the samples of the low node
    unsigned int cnt = 0;
    for (unsigned int i=0; i<nNodeSamples; i++) {
        if ( X[selectedSamples[i]][n->featureInd] <= optFeatureSplitValue)
            cnt++;
    }

    vector<int> lowList(cnt);
    vector<int> hiList(nNodeSamples-cnt);

    unsigned int lowCnt = 0, hiCnt = 0;
    double lowMean = 0.0, hiMean = 0.0;
    for ( unsigned int i=0; i<nNodeSamples; i++ ) {
        if ( X[selectedSamples[i]][n->featureInd] <= optFeatureSplitValue ) {
            lowList[lowCnt] = n->trainSamples[i];
            lowMean += target[n->trainSamples[i]];
            lowCnt++;
        }
        else {
            hiList[hiCnt] = n->trainSamples[i];
            hiMean += target[n->trainSamples[i]];
            hiCnt++;
        }
    }
    lowMean /= lowCnt;
    hiMean /= hiCnt;

    if ( hiCnt+lowCnt != nNodeSamples || lowCnt != cnt )
        assert ( false );

    // break, if too less samples
    if ( lowCnt < 1 || hiCnt < 1 )
    {
        n->featureInd = -1;
        n->value = lowCnt < 1 ? hiMean : lowMean;
        n->smallChild = nullptr;
        n->largeChild = nullptr;
        // todo: ???
        //if ( n->trainSamples )
        //delete[] n->trainSamples;
        n->trainSamples = vector<int>(0);

        nodeReduced currentNode;
        currentNode.pNode = n;
        currentNode.size = 0; // yiqian: size = 0, it will not be used again.
        nodesToSplit.push_back (currentNode);
        push_heap (nodesToSplit.begin(), nodesToSplit.end(), compareNodeReduced);

        return;
    }

    // prepare first new node
    node smallChild, largeChild;
    n->smallChild = &smallChild;
    n->smallChild->featureInd = -1; // yiqian: !!! new node, the featureNr is set to -1, so leaf->featureNr = -1
    n->smallChild->value = lowMean;
    n->smallChild->smallChild = nullptr; // yiqian: !!! child is set to 0, not nullptr
    n->smallChild->largeChild = nullptr;
    n->smallChild->trainSamples = lowList;

    // prepare second new node
    n->largeChild = &largeChild;
    n->largeChild->featureInd = -1;
    n->largeChild->value = hiMean;
    n->largeChild->smallChild = nullptr;
    n->largeChild->largeChild = nullptr;
    n->largeChild->trainSamples = hiList;

    // add the new two nodes to the heap
    nodeReduced lowNode, hiNode;
    lowNode.pNode = n->smallChild;
    lowNode.size = lowCnt;
    hiNode.pNode = n->largeChild;
    hiNode.size = hiCnt;

    nodesToSplit.push_back ( lowNode );
    push_heap ( nodesToSplit.begin(), nodesToSplit.end(), compareNodeReduced );

    nodesToSplit.push_back ( hiNode );
    push_heap ( nodesToSplit.begin(), nodesToSplit.end(), compareNodeReduced );
}


void GBDT::trainGBDT ( vector<vector<double>> const &X, vector<double> target)
{

    unsigned int nSamples = X.size();
    unsigned int nFeatures = X[0].size();

    for ( unsigned int epoch=0; epoch < nEpochs; epoch++ )
    {
        // train the model here

        vector<bool> usedFeatures (nFeatures,false);
        deque<nodeReduced> nodesToSplit;

        // root node has all examples in the training list
        node root;
        node* newTree = &root;

        newTree->featureInd = -1;
        newTree->value = 1e10;

        vector<int>range(nSamples);
        iota(range.begin(),range.end(), 0);
        newTree->trainSamples = range;

        nodeReduced firstNodeReduced;
        firstNodeReduced.pNode = newTree;
        firstNodeReduced.size = nSamples;


        nodesToSplit.push_back ( firstNodeReduced );
        push_heap ( nodesToSplit.begin(), nodesToSplit.end(), compareNodeReduced );

        // train the tree loop wise
        // call trainSingleTree recursive for the largest node
        for ( unsigned int j=0; j<maxTreeLeaves; j++ )
        {
            node* largestNode = nodesToSplit[0].pNode;
            trainSingleTree(largestNode, nodesToSplit
                            , X
                            , target
                            , usedFeatures);
        }
        trees.push_back(newTree);
        for ( size_t i = 0; i<X.size(); i++) {
            double p = predictSingleTree(newTree, X[i]);
            double err = target[i] - lRate*p;
            target[i] = err;
        }
    }
}

double GBDT::predictGBDT(vector<double> x) {
    double res=0;
    for (size_t i = 0; i<trees.size(); i++) {
        res += lRate * predictSingleTree(trees[i], x);
    }
    return res;
}


void GBDT::cleanTree(node *n) {

}




