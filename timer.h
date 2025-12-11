#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <mutex>
#include <iomanip>

class TimerAggregator {
private:
    static std::unordered_map<std::string, double> totals_;
    static std::unordered_map<std::string, int> counts_;
    static std::mutex mutex_;

public:
    static void add(const std::string& name, double microseconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        totals_[name] += microseconds;
        counts_[name]++;
    }

    static void printSummary() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\n===== TRAINING TIMING SUMMARY =====" << std::endl;
        std::cout << std::left << std::setw(25) << "Operation" 
                  << std::right << std::setw(12) << "Total (ms)" 
                  << std::setw(8) << "Count" 
                  << std::setw(12) << "Avg (μs)" << std::endl;
        std::cout << std::string(57, '-') << std::endl;
        
        for (const auto& pair : totals_) {
            const std::string& name = pair.first;
            double total_ms = pair.second / 1000.0;
            int count = counts_[name];
            double avg_us = pair.second / count;
            
            std::cout << std::left << std::setw(25) << name
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) << total_ms
                      << std::setw(8) << count
                      << std::setw(12) << std::fixed << std::setprecision(1) << avg_us << std::endl;
        }
        std::cout << std::string(57, '=') << std::endl;
    }
};

class SilentTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;
    
public:
    SilentTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~SilentTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        double microseconds = duration.count();
        
        TimerAggregator::add(name_, microseconds);
    }
};

// Simple macro for easy use
#define TIMING_SCOPE(name) SilentTimer timer(name)

#endif