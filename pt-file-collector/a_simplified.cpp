#include <iostream>
#include <string>
#include <stack>

using namespace std;

bool isValidList(const string& s) {
    string str = s;
    
    // 1. 基本检查：非空，以[开头，以]结尾
    if (str.empty()) return false;
    
    // 去除前后空格
    size_t start = str.find_first_not_of(" ");
    if (start == string::npos) return false;
    size_t end = str.find_last_not_of(" ");
    str = str.substr(start, end - start + 1);
    
    if (str[0] != '[' || str.back() != ']') return false;
    
    // 2. 使用栈来匹配括号和处理元素
    stack<int> brackets;  // 存储左括号的位置
    bool inNumber = false;  // 是否正在处理数字
    bool expectValue = true;  // 期望下一个是值而不是逗号
    
    for (size_t i = 0; i < str.size(); i++) {
        char c = str[i];
        
        // 处理空格
        if (c == ' ') {
            // 数字间的空格是非法的
            if (inNumber && i+1 < str.size() && 
                str[i+1] >= '0' && str[i+1] <= '9') {
                return false;
            }
            continue;  // 其他空格跳过
        }
        
        // 开始新的数字
        if (c >= '0' && c <= '9') {
            // 检查前导零
            if (c == '0' && i+1 < str.size() && 
                str[i+1] >= '0' && str[i+1] <= '9') {
                return false;
            }
            
            if (!inNumber && !expectValue) {
                return false;  // 例如 "][" 或 "1["
            }
            
            inNumber = true;
            expectValue = false;
            continue;
        }
        
        // 结束当前数字
        if (inNumber) {
            inNumber = false;
        }
        
        // 处理括号和逗号
        if (c == '[') {
            if (!expectValue) {
                return false;  // 例如 "1["
            }
            expectValue = true;
            brackets.push(i);
        } else if (c == ']') {
            // 检查是否有匹配的左括号
            if (brackets.empty()) {
                return false;
            }
            
            // 检查空列表或尾随逗号
            int openPos = brackets.top();
            brackets.pop();
            
            // 检查 [] 之间的内容
            if (i == openPos + 1) {
                // 空列表，合法
            } else if (str[i-1] == ',') {
                return false;  // 尾随逗号
            }
            
            expectValue = false;
        } else if (c == ',') {
            if (expectValue || i == str.size() - 1) {
                return false;  // 连续逗号或末尾逗号
            }
            expectValue = true;
        } else {
            // 其他字符非法
            return false;
        }
    }
    
    // 检查所有括号是否匹配
    return brackets.empty();
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
