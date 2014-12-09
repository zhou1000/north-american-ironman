#include <cstring>
#include <stdexcept>
#include <iostream>

#include "DataReader.h"

using namespace std;

unsigned int DataReader::countLines(string const &path) {
    ifstream f(path, ifstream::in);
    if (f.is_open() == false) {
        throw runtime_error(string("cannot open file: ") + path);
    }
    unsigned int nLines = 0;
    string line;
    while (getline(f, line)) {
        ++nLines;
    }
    f.close();
    return nLines;
}

unsigned int DataReader::countFields(string const &path, char delimiter) {
    ifstream f(path, ifstream::in);
    if (f.is_open() == false) {
        throw runtime_error(string("cannot open file: ") + path);
    }
    string line;
    getline(f,line);
    stringstream lineStream(line);

    unsigned int nFields = 0;
    string cell;

    while(getline(lineStream,cell,delimiter)) {
        ++ nFields;
    }
    f.close();
    return nFields;
}

vector<vector<double>> DataReader::readCSVdata(string const &path, int skipLine, char delimiter) {

    unsigned int nRows = countLines(path) - skipLine;
  

    unsigned int nColumns = countFields(path, delimiter);
    vector<vector<double>> data(nRows, vector<double>(nColumns));

    ifstream f(path, ifstream::in);
    string line, cell;
    for (int i = 0; i<skipLine; ++i) {
        getline(f, line);
    }
    for (unsigned int i = 0; i<nRows - skipLine; ++i) {
        getline(f, line);
        stringstream lineStream(line);
        for (unsigned int j = 0; j<nColumns; ++j) {
            getline(lineStream, cell,delimiter);
            data[i][j] = atof(cell.c_str());
        }
    }
    f.close();
    return data;
}

vector<string> DataReader::readCSVheader(string const &path, int headerRow, char delimiter) {
    unsigned int nColumns = countFields(path, delimiter);
    vector<string> header(nColumns);

    ifstream f(path, ifstream::in);
    string line;
    for (int i = 0; i<headerRow; ++i) {
        getline(f, line);
    }
    getline(f, line);
    stringstream lineStream(line);
    for (unsigned int j = 0; j<nColumns; ++j) {
        getline(lineStream, header[j], delimiter);
    }
    f.close();
    return header;
}




