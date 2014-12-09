#include <stdexcept>
#include <fstream>

#include "DataWriter.h"

using namespace std;

void DataWriter::writeCSV(string const &path, vector<string> const &header
                          , vector<vector<double>> const &data, char delimiter) {
    ofstream f(path);
    if (f.is_open() == false) {
        throw runtime_error(string("cannot open file: ") + path);
    }
    size_t j = 0;
    for (; j<header.size()-1; ++j) {
        f<<header[j]<<to_string(delimiter);
    }
    f<<header[j]<<endl;
    for (size_t i=0; i<data.size(); ++i) {
        j = 0;
        for (j=0; j<header.size()-1; ++j) {
            f<<to_string(data[i][j])<<to_string(delimiter);
        }
        f<<to_string(data[i][j])<<endl;;
    }
    f.close();
}

