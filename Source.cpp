/*
This my suffix tree implementation done for Algorithms-2 course 
(homeworks 2-3 and 2-4).

I tried not to use brain too much and code as close to Maxim Babenko in his
lectures as possible.

Still mistaken quitee a lot and needed to test all about =(
*/


#include <string>
#include <vector>
#include <map>
#include <deque>
#include <iostream>
#include <fstream>
#include <cstdint>

class SuffixSubstring {
public:
  int begin;
  int end;

  static const int kInfEnd = -1;

  int length() {
    return end - begin;
  }

  bool is_endless() {
    return end == kInfEnd;
  }
};

struct SuffixTreeNode {
  std::map<char, SuffixTreeNode*> children;
  SuffixSubstring edge = {0, -1};
  SuffixTreeNode* suff_link = NULL;
};

struct SuffixTreeLocation {
  SuffixTreeNode* prev_node;
  SuffixTreeNode* next_node;
  int edge_symbols;
};

class SuffixTree {
public:
  using Node = SuffixTreeNode;
  using Location = SuffixTreeLocation;

  explicit SuffixTree(const std::string& str_in) 
    : str_(str_in) {
    root_ = new Node();
    root_->suff_link = root_;
    root_->edge = {0, 0};
    insert_loc_ = {root_, root_, 0};

    int size = str_.size();
    while (symb_read_ < size) {
      MakeIteration();
      ++symb_read_;
      //  PrintTree();
    }
  }

  void PrintTree() {
    std::deque<Node*> order;
    order.push_back(root_);

    while (!order.empty()) {
      Node* node = order.front();

      int begin = node->edge.begin;
      int end = node->edge.end == -1 ? symb_read_ : node->edge.end;

      std::cout << "Node with edge " << PrintNode(node)
        << " has childs: ";
      if (node->children.empty()) {
        std::cout << "none";
      }

      for (auto& pair : node->children) {
        Node* child = pair.second;
        order.push_back(child);
        std::cout << pair.first << " ";
      }
      
      order.pop_front();
      std::cout << '\n';
    }
    std::cout << "insert location: "
      << PrintNode(insert_loc_.prev_node) << " "
      << PrintNode(insert_loc_.next_node) << " "
      << insert_loc_.edge_symbols <<'\n';
    std::cout << std::endl;
  }

  int64_t CountSubstrings() {
    int64_t count = 0;
    std::deque<Node*> order;
    order.push_back(root_);

    while (!order.empty()) {
      Node* node = order.front();
      int begin = node->edge.begin;
      int end = node->edge.end == -1 ? str_.size() : node->edge.end;
      count += end - begin;
      for (auto& pair : node->children) {
        Node* child = pair.second;
        order.push_back(child);
      }
      order.pop_front();
    }
    return count;
  }

private:
  bool IsLeaf(const Location& loc) {
    int depth = symb_read_ - loc.next_node->edge.begin + 1 - loc.edge_symbols;
    return loc.next_node->edge.is_endless() && !depth;
  }

  bool IsRealVertex(const Location& loc) {
    return loc.edge_symbols == 0;
  }

  bool IsRoot(const Location& loc) {
    return loc.prev_node == root_ && loc.edge_symbols == 0;
  }

  char FirstEdgeSymbol(const Location& loc) {
    return str_[loc.next_node->edge.begin];
  }

  std::string PrintNode(Node* node) {
    int begin = node->edge.begin;
    int end = node->edge.end == -1 ? symb_read_ : node->edge.end;
    return std::to_string(begin) + " "
      + std::to_string(end) + " " + str_.substr(begin, end - begin);
  }

  SuffixSubstring Path(const Location& loc) {
    int begin = loc.next_node->edge.begin;
    return {begin, begin + loc.edge_symbols};
  }

  Location SkipAndCount(Node* start, SuffixSubstring path) {
    Node* next = start;
    if (path.length()) {
      next = start->children[str_[path.begin]];
    } 
    while (!next->edge.is_endless() && next->edge.length() <= path.length()) {
      path.begin += next->edge.length();
      start = next;
      if (path.length()) {
        next = start->children[str_[path.begin]];
      } else {
        break;
      }
    }
    return {start, next, path.length()};
  }

  Location SuffixLink(const Location& loc) {
    Node* ancestor = loc.prev_node->suff_link;
    if (IsRealVertex(loc)) {
      return {ancestor, ancestor, 0};
    } else {
      SuffixSubstring path = Path(loc);
      if (loc.prev_node == root_) {
        path.begin += 1;
      } 
      return SkipAndCount(ancestor, path);
    }
  }

  bool HasSymbolInContext(const Location& loc, char symb) {
    if (IsRealVertex(loc)) {
      return loc.prev_node->children.count(symb) > 0;
    } else {
      int pos = loc.next_node->edge.begin + loc.edge_symbols;
      return str_[pos] == symb;
    }
  }

  Node* AddChild(Location& loc, int right) {
    Node* child = new Node;
    child->edge = {right, -1};
    char symb = str_[right];
    if (IsRealVertex(loc)) {
      loc.prev_node->children[symb] = child;
      loc.next_node = child;
      return NULL;
    } else {
      
      int edge_begin = loc.next_node->edge.begin;
      int edge_mid = edge_begin + loc.edge_symbols;

      Node* mid_node = new Node;
      loc.prev_node->children[str_[edge_begin]] = mid_node;

      mid_node->edge = {edge_begin, edge_mid};
      mid_node->children[str_[edge_mid]] = loc.next_node;
      mid_node->children[symb] = child;
      // mid_node->suff_link = ?

      loc.next_node->edge.begin += loc.edge_symbols;
      loc.next_node = mid_node;
      return mid_node;
    }
  }

  void MakeIteration() {
    char symb = str_[symb_read_];
    std::vector<Node*> new_node_memory(2, NULL);

    while (!HasSymbolInContext(insert_loc_, symb)) {
      new_node_memory[1] = AddChild(insert_loc_, symb_read_);
      if (new_node_memory[0])  {
        new_node_memory[0]->suff_link = new_node_memory[1] ?
          new_node_memory[1] : insert_loc_.prev_node;        
      }
      new_node_memory[0] = new_node_memory[1];
      insert_loc_ = SuffixLink(insert_loc_);
    }

    if (insert_loc_.edge_symbols == 0) {
      insert_loc_.next_node = insert_loc_.prev_node->children[symb];
    }

    if (new_node_memory[0]) {
      new_node_memory[0]->suff_link = insert_loc_.prev_node; 
    }

    ++insert_loc_.edge_symbols;
    if (insert_loc_.edge_symbols == insert_loc_.next_node->edge.length()) {
      insert_loc_ = {insert_loc_.next_node, insert_loc_.next_node, 0};
    }

    while (IsLeaf(insert_loc_) && !IsRoot(insert_loc_)) {
      insert_loc_ = SuffixLink(insert_loc_);
    }
  }

  Node* root_;
  Location insert_loc_;
  std::string str_;
  int symb_read_ = 0;
};

std::string ReadString() {
  std::string str;
  std::cin >> str;
  return str;
}

void WriteAnswer(int64_t answer) {
  std::cout << answer;
}

void UnitTest();
void StressTest();

int64_t CountSubstringsSlow(const std::string& str) {
  int size = str.size();
  std::map<std::string, int> counter;
  
  for (int left = 0; left <= size; ++left) {
    for (int right = left; right <= size; ++right) {
      counter[str.substr(left, right - left)] += 1;
    }
  }
  return counter.size();
}

int main() {
  bool test = false;
  if (test) {
    std::ofstream out("input.txt");
    std::cout.rdbuf(out.rdbuf());

    //  UnitTest();
    StressTest();
  } else {
    std::string str = ReadString();
    SuffixTree tree(str);
    int64_t ans = tree.CountSubstrings();
    WriteAnswer(ans);
  }
  
  return 0;
}

template<typename U, typename V, typename W>
void CompareResults(U test, V excpected, W result) {
  if (excpected != result) {
    std::cout << "On test " << test
      << " excpected " << excpected
      << " but got " << result << std::endl;
  }
}

void UnitTest() {
  std::string str = "ababaab";
  SuffixTree tree(str);
  //  tree.PrintTree();
  int64_t ans = tree.CountSubstrings();
  int64_t ans_true = CountSubstringsSlow(str) - 1;
  CompareResults(str, ans_true, ans);
  std::cout << std::endl;
}

bool TestCountSubstrings(const std::string& str) {
  SuffixTree tree(str);
  int64_t ans = tree.CountSubstrings();
  int64_t ans_true = CountSubstringsSlow(str) - 1;
  CompareResults(str, ans_true, ans);
  return ans == ans_true;
}

void StressTest() {
  int size_min = 110;
  int size_max = 120;
  int max_letter = 10;

  for (int size = size_min; size <= size_max; ++size) {
    srand(123);
      
    std::string test;
    for (int letter = 0; letter < size; ++letter) {
      char rnd = std::rand() % max_letter + 'a';
      test.push_back(rnd);
    }

    if (!TestCountSubstrings(test)) break;
    std::cout << "size " << size << " checked" << std::endl;
  }
}
