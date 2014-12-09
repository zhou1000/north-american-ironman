// Yiqian, 20141122
//  FTRL algorithm: Follow the regularized leader - proximal
//  Reference:
//  http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf

class FTRL{
    public:
        FTRL(unsigned long);
        ~FTRL()=default;
        double PredictOne(const std::vector<int> &x);
        void TrainOne(const std::vector<int> &x, const double y);
        void SaveModel(const std::string&);
        void LoadModel(const std::string&);
        inline unsigned long ModelSize () {return w.size();}
   private:
        double alpha;
        double beta;
        double l1;
        double l2;

        std::vector<double> n;
        std::vector<double> z;
        std::vector<double> w;
};

