#include <iostream>
#include <string>

using namespace std;

// 前向声明
bool isInteger(const string& s);
bool isValidElement(const string& elem, int depth = 0);
bool isValidList(const string& s, int depth = 0);

// 函数：自定义字符判断函数
bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

// 函数：检查字符串是否为整数
bool isInteger(const string& s) {
    // 去除前后空格
    size_t start = s.find_first_not_of(" ");
    size_t end = s.find_last_not_of(" ");
    if (start == string::npos) return false;
    string trimmed = s.substr(start, end - start + 1);
    
    if (trimmed.empty()) return false;
    
    // 检查前导零
    if (trimmed.size() > 1 && trimmed[0] == '0') {
        return false;  // 不允许前导零
    }
    
    // 检查数字之间是否有空格以及是否只包含数字
    bool hasDigit = false;
    for (size_t i = 0; i < trimmed.size(); i++) {
        if (isDigit(trimmed[i])) {
            hasDigit = true;
        } else if (trimmed[i] == ' ') {
            // 如果空格前后都是数字，则非法
            if (i > 0 && i < trimmed.size()-1 && 
                isDigit(trimmed[i-1]) && isDigit(trimmed[i+1])) {
                return false;
            }
        } else {
            return false; // 其他非法字符
        }
    }
    
    return hasDigit; // 确保至少有一个数字
}

// 函数：检查单个元素是否合法（整数或嵌套列表）
bool isValidElement(const string& elem, int depth) {
    // 避免过深递归
    const int MAX_DEPTH = 100;
    if (depth > MAX_DEPTH) return false;
    
    // 去除前后空格
    size_t start = elem.find_first_not_of(" ");
    size_t end = elem.find_last_not_of(" ");
    if (start == string::npos) return false;
    string trimmed = elem.substr(start, end - start + 1);
    
    // 检查是否为整数
    if (isInteger(trimmed)) return true;
    
    // 检查是否为嵌套列表
    return isValidList(trimmed, depth + 1);
}

// 函数：检查字符串是否为合法的列表
bool isValidList(const string& s, int depth) {
    // 避免过深递归
    const int MAX_DEPTH = 100;
    if (depth > MAX_DEPTH) return false;
    
    // 去掉字符串两端的空格
    size_t start = s.find_first_not_of(" ");
    size_t end = s.find_last_not_of(" ");
    if (start == string::npos) return false;
    string trimmed = s.substr(start, end - start + 1);
    
    // 检查是否以 '[' 开头和 ']' 结尾
    if (trimmed.empty() || trimmed[0] != '[' || trimmed.back() != ']') return false;
    
    // 去掉最外层的 '[' 和 ']'
    string inner = trimmed.substr(1, trimmed.size() - 2);
    
    // 如果内部为空，直接返回 true（空列表是合法的）
    if (inner.find_first_not_of(" ") == string::npos) return true;
    
    // 处理元素
    string current;
    int bracketLevel = 0;
    bool expectElement = true; // 期望下一个是元素而不是逗号
    
    for (size_t i = 0; i < inner.length(); i++) {
        char ch = inner[i];
        
        if (ch == '[') {
            bracketLevel++;
            current += ch;
            expectElement = false;
        } else if (ch == ']') {
            bracketLevel--;
            if (bracketLevel < 0) return false; // 括号不匹配
            current += ch;
            expectElement = false;
        } else if (ch == ',' && bracketLevel == 0) {
            // 只有在最外层才分割元素
            
            // 检查是否为空元素（连续逗号或开头逗号）
            if (expectElement || current.find_first_not_of(" ") == string::npos) {
                return false; // 发现空元素，格式非法
            }
            
            // 验证当前元素
            if (!isValidElement(current, depth)) {
                return false;
            }
            
            current = "";
            expectElement = true; // 逗号后必须是元素
        } else if (ch == ' ') {
            // 空格，只添加到当前元素
            current += ch;
        } else if (isDigit(ch) || bracketLevel > 0) {
            // 数字或者在嵌套列表内的任何字符
            current += ch;
            expectElement = false;
        } else {
            // 在最外层出现的非法字符
            return false;
        }
    }
    
    // 处理最后一个元素
    if (current.find_first_not_of(" ") == string::npos) {
        // 如果最后一个元素是空的，则非法
        return false;
    }
    
    // 检查最后一个元素是否合法
    if (!isValidElement(current, depth)) {
        return false;
    }
    
    if (bracketLevel != 0) return false; // 括号不匹配
    
    return true;
}

int main() {
    string input;
    getline(cin, input);
    
    if (isValidList(input)) {
        cout << "格式合法！" << endl;
    } else {
        cout << "格式非法！" << endl;
    }
    
    return 0;
}