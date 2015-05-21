#ifndef STOP_H
#define STOP_H

#include <chrono>

class Stopwatch
{
public:
    Stopwatch()
    {
        oldTime = std::chrono::high_resolution_clock::now();
        newTime = oldTime;
    }

    void tick()
    {
        oldTime = newTime;
        newTime = std::chrono::high_resolution_clock::now();
    }

    // Returns time elapsed in seconds
    float Elapsed() const
    {
        using std::chrono::duration_cast;
        using std::chrono::duration;
        return (duration_cast<duration<float>>(newTime - oldTime)).count();
    }

private:
    std::chrono::high_resolution_clock::time_point newTime, oldTime;
};


#endif // STOP_H