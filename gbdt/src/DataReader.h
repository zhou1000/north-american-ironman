#ifndef __DATAREADER_
#define __DATAREADER_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>



class DataReader {
    public:
        unsigned int countLines(std::string const &path);
        unsigned int countFields(std::string const &path, char delimiter = ',');
        
        std::vector<std::vector<double> > readCSVdata(std::string const &path, int skipLine=1, char delimiter = ',');
        std::vector<std::string> readCSVheader(std::string const &path, int headerRow = 0, char delimiter = ',');

};

#endif


