#include "timer.h"

// Static member definitions
std::unordered_map<std::string, double> TimerAggregator::totals_;
std::unordered_map<std::string, int> TimerAggregator::counts_;
std::mutex TimerAggregator::mutex_;
