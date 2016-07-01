#include <forward_list>
#include <stack>
#include <iostream>

template <typename T>
void Relink(std::forward_list<T>* flist) {
  size_t size = std::distance(flist->begin(), flist->end());
  if (size <= 2) return;

  auto middle = std::next(flist->before_begin(), size - size / 2);
  std::stack<T> latter_half;
  while (std::next(middle) != flist->end()) {
    latter_half.push(*std::next(middle));
    flist->erase_after(middle);
  }

  for (auto it = flist->begin(); !latter_half.empty(); ++it) {
    it = flist->insert_after(it, latter_half.top());
    latter_half.pop();
  }
}

void TestAll();

int main() {
  TestAll();
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::forward_list<T>& flist) {
  os << "[";
  for (const auto& item : flist) {
    os << item << " ";
  }
  return os << "]";
}

template <typename U, typename V, typename W>
void CheckResult(U input, V result, W excpected) {
  if (result != excpected) {
    std::cout << "for test: " << input 
      << "got result: " << result 
      << "but expected: " << excpected;
  }
}

template <typename T>
void UnitTestRelink(const std::forward_list<T>& test, const std::forward_list<T>& answer) {
  std::forward_list<T> test_copy(test);
  Relink(&test_copy);
  CheckResult(test, test_copy, answer);
}

void TestAll() {
  UnitTestRelink<int>({1}, {1});
  UnitTestRelink<int>({1, 2}, {1, 2});
  UnitTestRelink<int>({1, 2, 3, 4}, {1, 4, 2, 3});
  UnitTestRelink<int>({1, 2, 3, 4, 5}, {1, 5, 2, 4, 3});
}
