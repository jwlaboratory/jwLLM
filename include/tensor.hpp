#pragma once

#include <vector>

class Tensor
{
public:
    std::vector<int> dimension_sizes;
    std::vector<float> data;

    Tensor(std::vector<int> dims, std::vector<float> passed_data);
};
