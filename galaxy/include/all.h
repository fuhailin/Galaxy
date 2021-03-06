#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/variant.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "eigen_func.h"
#include "hash_method.h"
#include "nlohmann/json.hpp"
#include "ps/ps.h"
#include "tools.h"
#include <fstream>
#include "glog/logging.h"
#include "gflags/gflags.h"

namespace galaxy {
typedef uint64_t ull;
using SLOT_ID_FEAS = std::pair<int, std::vector<std::string>>;

inline nlohmann::json &global_conf() {
  static nlohmann::json conf;
  return conf;
}

class Layer {};
} // namespace galaxy
