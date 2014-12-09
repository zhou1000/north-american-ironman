#ifndef __GBDT_
#define __GBDT_

// based on ELF project
// http://elf-project.sourceforge.net/
// Yiqian

#include <vector>
#include <deque>

/**
 * This is one node of the tree
 */
struct node {
    unsigned int featureInd;
    double value;
    node* smallChild;
    node* largeChild;
    std::vector<int> trainSamples; // index (row nubmer) of training sample for this nodes
};

/**
 * this struct is used to build the heap data structure (for selecting the largest node)
 */
struct nodeReduced {
    node* pNode;
    unsigned int size;
}; // only record the pointer and size of sample. it's used in deque.


class GBDT {
    public:
        GBDT();
        ~GBDT();

        void trainGBDT (std::vector<std::vector<double>> const &X, std::vector<double> target);
        double predictGBDT(std::vector<double> x);

    private:
        void trainSingleTree(node* n, std::deque<nodeReduced> &nodesToSplit
                             , std::vector<std::vector<double>> const &X
                             , std::vector<double> const &y
                             , std::vector<bool> usedFeatures);
        double predictSingleTree(node* root, std::vector<double> singleSample);
        void cleanTree(node *n);

        unsigned int maxTreeLeaves;
        unsigned int featureSubspaceSize;
        unsigned int nEpochs;
        bool doOptSplit;
        std::vector<node*> trees;
        double lRate;
};

#endif





