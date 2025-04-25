#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <iomanip>

using namespace std;

// 修改声明以匹配a.cpp中的实现
extern bool isInteger(const string& s);
extern bool isValidElement(const string& elem, int depth = 0);
extern bool isValidList(const string& s, int depth = 0);

// 运行测试的函数
void run_tests() {
    // 测试用例数组
    vector<pair<string, bool>> testCases = {
        {"[1, 2, [3, 4]]", true},      // 题目给的合法例子
        {"[1, [2], 3 ]", true},        // 题目给的合法例子
        {"[[1]]", true},               // 题目给的合法例子
        {"[12 34]", false},            // 题目给的非法例子
        {"[]", true},                  // 空列表
        {"[1,2,3]", true},             // 无空格版本
        {"[1, 2, 3]", true},           // 有空格版本
        {"[1,[2,3],4]", true},         // 多层嵌套
        {"[1, [2, [3]], 4]", true},    // 更复杂的嵌套
        {"[1, 2", false},              // 缺少右括号
        {"1, 2, 3]", false},           // 缺少左括号
        {"[1, 2, 3", false},           // 缺少右括号
        {"[1, 2,,3]", false},          // 多余的逗号
        {"[1, 2, 3,]", false},         // 末尾多余的逗号
        {"[,1, 2, 3]", false},         // 开头多余的逗号
        {"[1 2 3]", false},            // 缺少逗号
        {"[[1, 2], [3, 4]", false},    // 嵌套列表缺少右括号
        {"[1, 2, [3, 4]", false},      // 嵌套列表缺少右括号
        
        // 新增测试用例
        {"[[1, 2, 3], [4, 5, 6]], [7, 8, 9]]", false},  // 非法：额外的右括号
        {"[[1, 2, 3], [4, 5, 6], [7, [8, 9, 10]], []]", true}, // 合法：嵌套列表，含空列表
        {"[, 2]", false},              // 非法：开头有逗号
        
        // 新增更严格的测试用例
        {"[1.23, 4]", false},          // 非法：浮点数
        {"[\"happy\", 1]", false},     // 非法：字符串
        {"[1-2, 3]", false},           // 非法：表达式
        {"[[[[[1]]]]]", true},         // 合法：多级嵌套
        {"[01, 2]", false},            // 非法：前导零
        {"[-1, 2]", false},            // 非法：负数
        {"[1+2, 3]", false},           // 非法：数学表达式
        {"[   1   ,   2   ]", true},   // 合法：超多空格
        {"[[], [1], [1, [2, [3]]]]", true}, // 合法：复杂嵌套组合
        {"[[], [], []]", true},        // 合法：多个空列表
        {"[123456789, 2]", true},      // 合法：大整数(在int范围内)
        {"[ 1, 2, 3 ]", true},         // 合法：Python风格空格
        {"[1,,]", false},              // 非法：连续逗号和尾随逗号
        
        // 更多边缘情况测试
        {"", false},                   // 非法：空字符串
        {"[", false},                  // 非法：只有左括号
        {"]", false},                  // 非法：只有右括号
        {"[][]", false},               // 非法：连续列表无分隔
        {"[[]", false},                // 非法：嵌套列表不完整
        {"[1 [2]]", false},            // 非法：缺少逗号
        {"[[1] 2]", false},            // 非法：缺少逗号
        {"[1\t,2]", false},            // 非法：题目只说明空格是合法的，其他空白字符未提及
        {"[1\n,2]", false},            // 非法：题目只说明空格是合法的，其他空白字符未提及
        {"[1, 2, 3, ]", false},        // 非法：尾随逗号
        {"[1, 2, 3,]", false},         // 非法：尾随逗号
        {"[[],]", false},              // 非法：尾随逗号
        {"[1,[2,],3]", false},         // 非法：嵌套列表中有尾随逗号
        {"[ [1] ]", true},             // 合法：嵌套列表周围有空格
        {"[1, 2, 3,[]]", true},        // 合法：包含空列表元素
        {"[9223372036854775807]", true}, // 合法：64位整数最大值
        {"[2147483647]", true},        // 合法：32位整数最大值
        {"[0, 1, 2]", true},           // 合法：包含0
        {"[000]", false},              // 非法：多个前导零
        {"[1,,2]", false},             // 非法：连续逗号
        {"[,]", false},                // 非法：只有一个逗号
        {"[1 + 2]", false},            // 非法：包含表达式
        {"[true]", false},             // 非法：布尔值
        {"[null]", false}              // 非法：空值
    };
    
    cout << "============ 列表格式验证测试 ============" << endl;
    cout << left << setw(25) << "输入" << " | " << "预期" << " | " << "实际" << " | " << "结果" << endl;
    cout << string(60, '-') << endl;
    
    int passedTests = 0;
    
    for (const auto& [input, expected] : testCases) {
        bool result = isValidList(input);
        string expectedStr = expected ? "合法" : "非法";
        string resultStr = result ? "合法" : "非法";
        string statusStr = (result == expected) ? "通过" : "失败";
        
        cout << left << setw(25) << input << " | ";
        cout << left << setw(4) << expectedStr << " | ";
        cout << left << setw(4) << resultStr << " | ";
        cout << statusStr << endl;
        
        if (result == expected) {
            passedTests++;
        }
    }
    
    cout << string(60, '-') << endl;
    cout << "测试结果：" << passedTests << "/" << testCases.size() << " 通过 (";
    cout << fixed << setprecision(1) << (passedTests * 100.0 / testCases.size()) << "%)" << endl;
}

#ifdef TESTING
// 自动执行测试的初始化器
class TestRunner {
public:
    TestRunner() {
        run_tests();
    }
};

// 全局对象，程序启动时自动运行测试
static TestRunner test_runner;
#endif
