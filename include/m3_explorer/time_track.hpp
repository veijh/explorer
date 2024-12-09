#ifndef TIME_TRACK_HPP
#define TIME_TRACK_HPP
#include <chrono>
#include <iostream>

class TimeTrack {
private:
  std::chrono::system_clock::time_point start_time_;

public:
  TimeTrack() : start_time_(std::chrono::system_clock::now()){};
  void SetStartTime() { start_time_ = std::chrono::system_clock::now(); }
  void OutputPassingTime(const std::string &item_name) {
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end_time - start_time_;
    std::cout << "[" << item_name << "]: " << elapsed.count() << " ms"
              << std::endl;
  }
};

#endif
