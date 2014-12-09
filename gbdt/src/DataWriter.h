#include <vector>
#include <string>

class DataWriter {
    public:
        void writeCSV(std::string const &path, std::vector<std::string> const &header
                      , std::vector<std::vector<double>> const &data, char delimiter = ',');
};

