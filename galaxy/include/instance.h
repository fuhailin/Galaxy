#pragma once
#include "all.h"

namespace galaxy {
    struct Instance {
        int label;
        std::vector<ull> feas;
        std::vector<double> vals;
        std::vector<SLOT_ID_FEAS> slot_feas;
        float pre;
    };
}