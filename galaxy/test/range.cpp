#include <generator.h>
#include <iostream>

galaxy::Generator<int> range(int max){
  return galaxy::Generator<int>([=](galaxy::Yield<int> &yield){
    for(int i = 0;i<max;++i) yield(i);
  });
}

int main(){
  for(int i:range(10)) std::cout << i << std::endl;
}

