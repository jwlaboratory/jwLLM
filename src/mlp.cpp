#include "mlp.hpp"
#include <stdexcept>

MLP::MLP(SafeTensors &weights, const std::string &prefix)
{
    FC_WEIGHTS = weights.get(prefix + "mlp.c_fc.weight");
    FC_BIAS = weights.get(prefix + "mlp.c_fc.bias");
    PROJECTION_WEIGHTS = weights.get(prefix + "mlp.c_proj.weight");
    PROJECTION_BIAS = weights.get(prefix + "mlp.c_proj.bias");
}

Matrix MLP::forward(const Matrix &x) const
{
    //[seq, dmodel]

    // same idea as  low rank facotirzation But INVERSED. we project into big zone so we can seperate, then project done
    Matrix widened = x.multiply(FC_WEIGHTS).broadcast_add_row(FC_BIAS);

    widened = widened.gelu();

    Matrix shrunk = widened.multiply(PROJECTION_WEIGHTS).broadcast_add_row(PROJECTION_BIAS);

    return shrunk;

    // note. a LOT OF THE PARAMS live here. this is a HEAVY spot, very important
    // without non linearity this would be waste of time
}
