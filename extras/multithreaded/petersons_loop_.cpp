class Petersons
{

public:
    int turn = 0;
    bool desire[2];

    Petersons()
    {
        desire[0] = false;
        desire[1] = false;
    }

    void func1()
    {

        desire[0] = true;
        turn = 1;
        while (turn != 0 && desire[1])
        {
            ;
        }
        // do work

        // return
        desire[0] = 0;
    };

    void func2()
    {
        desire[1] = true;
        turn = 0;
        while (turn != 1 && desire[0])
        {
            ;
        }

        desire[1] = 0;
    }
};