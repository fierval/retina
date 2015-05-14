#include <chrono>
#pragma once

class Stopwatch
{
public:
    explicit Stopwatch(bool start_immediately = false);
    void Start(bool reset = false);
    void Stop();
    long long Elapsed() const;

private:
    chrono::time_point<std::chrono::system_clock> start, end;
    bool running;
};
Stopwatch::Stopwatch(bool start_immediately)
    : start(), end(), running(false)
{
    if (start_immediately)
    {
        Start(true);
    }
}
void Stopwatch::Start(bool reset)
{
    if (!running)
    {
        if (reset)
        {
            start = chrono::system_clock::now();
        }
        running = true;
    }
}
void Stopwatch::Stop()
{
    if (running)
    {
        end = chrono::system_clock::now();
        running = false;
    }
}
long long Stopwatch::Elapsed() const
{
    chrono::milliseconds elapsed(((running ? chrono::system_clock::now() : end) - start).count());
    return elapsed.count();
}