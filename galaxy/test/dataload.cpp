#include "dataload.h"
#include "tools.h"
#include <assert.h>
int main(){
    int epoch = 10;
    std::string path = "./data/test0";
    int batch_size = 9, line_len = galaxy::get_file_len(path);
    auto data_iter = galaxy::dataload(path, epoch, batch_size, false, 10);
    int cnt = 0;
    bool first = true;
    for (std::vector<galaxy::Instance> x : data_iter) {
        cnt += x.size();
        if (first) {
            assert(cnt == batch_size);
            first = false;
        }
        for (auto y : x)
            galaxy::print_vec("", y.feas);
    }
    assert(cnt == line_len * 10);
}