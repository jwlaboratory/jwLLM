#include <iostream>
using std::cout;
using std::endl;
using std::vector;

class Tensor
{
public:
    vector<int> dimension_sizes;
    vector<float> data;

    Tensor(vector<int> dims, vector<float> passed_data)
    {

        int max_total = dims[0];
        for (int i = 1; i < dims.size(); i++)
        {
            max_total *= dims[i];
        }
        if (max_total != passed_data.size())
        {
            throw std::invalid_argument("dimensions must match total amount of data");
        }

        dimension_sizes = dims;
        data = passed_data;
    }
};