// Copyright (c) 2019 Youlin Feng <fengyoulin@foxmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// #define EIGEN_USE_MKL_ALL

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "trainer.h"

int reserve_count = 0;

unsigned char from_digits(const char *c, int length)
{
    unsigned char b = 0;
    for (int i = 0; i < length && i < 3; ++i) {
        b *= 10;
        b += c[i] - '0';
    }
    return b;
}

std::string to_hex(unsigned char b)
{
    std::string s;
    char c[] = {char((b/16)&15), char(b&15)};
    for (int i = 0; i < 2; ++i) {
        char h = c[i];
        if (h <= 9) {
            h += '0';
        } else {
            h += 'A' - 10;
        }
        s += h;
    }
    return s;
}

std::pair<unsigned char, std::vector<unsigned char>> parse_line(std::string &s)
{
    unsigned t = 0;
    std::vector<unsigned char> v(784);
    size_t length = s.length();
    const char *p = s.c_str();
    const char *b = p, *e = p;
    while (e - p <= length) {
        if (*e == ',') {
            t = from_digits(b, e - b);
            b = ++e;
            break;
        }
        ++e;
    }
    int i = 0;
    while (e - p <= length) {
        if (e - p == length || *e == ',') {
            v[i++] = from_digits(b, e - b);
            b = e + 1;
        }
        ++e;
    }
    return std::make_pair(t, v);
}

int load_data(const char *path, std::vector<std::pair<unsigned char, std::vector<unsigned char>>> &data)
{
    std::ifstream is(path, std::ios::in);
    if (!is.is_open()) {
        std::cout << "cannot open: " << path << std::endl;
        return -1;
    }
    int i = 0;
    std::string line;
    while (std::getline(is, line)) {
        if (i >= reserve_count) {
            data.push_back(parse_line(line));
            ++i;
        } else {
            data[i++] = parse_line(line);
        }
    }
    is.close();
    return 0;
}

bool is_digits(std::string &s)
{
    for (auto c : s) {
        if (!std::isdigit(c)) {
            return false;
        }
    }
    return true;
}

void interact(std::vector<std::pair<unsigned char, std::vector<unsigned char>>> &data, trainer<784, 225, 10> &tr)
{
    std::string s;
    while (true) {
        std::cin >> s;
        if (s.length() > 0) {
            if (s == "q" ||
                s == "quit" ||
                s == "exit") {
                break;
            }
            if (s.substr(0, 4) == "load") {
                s = s.substr(4);
                while (!s.empty() && s[0] == ':') {
                    s = s.substr(1);
                }
                if (s.empty()) {
                    std::cout << "no file path" << std::endl;
                    continue;
                }
                if (tr.loadModel(s.c_str()) < 0) {
                    std::cout << "cannot load: " << s << std::endl;
                } else {
                    std::cout << "loaded" << std::endl;
                }
                continue;
            }
            if (s.substr(0, 4) == "save") {
                s = s.substr(4);
                while (!s.empty() && s[0] == ':') {
                    s = s.substr(1);
                }
                if (s.empty()) {
                    std::cout << "no file path" << std::endl;
                    continue;
                }
                if (tr.saveModel(s.c_str()) < 0) {
                    std::cout << "cannot save: " << s << std::endl;
                } else {
                    std::cout << "saved" << std::endl;
                }
                continue;
            }
            if (s == "train") {
                auto bt = std::chrono::system_clock::now();
                Matrix<float, Dynamic, 784> m = Matrix<float, Dynamic, 784>::Random(30, 784);
                Matrix<float, Dynamic, 10> t = Matrix<float, Dynamic, 10>::Random(30, 10);
                int i = 0;
                while (i < data.size()) {
                    int e = i + 30;
                    if (e > data.size()) {
                        e = data.size();
                    }
                    int b = i;
                    for (; i < e; ++i) {
                        const auto &p = data[i];
                        for (int j = 0; j < 784; ++j) {
                            m(i - b, j) = (p.second[j] * 0.999 / 255) + 0.001;
                        }
                        for (int j = 0; j < 10; ++j) {
                            if (j == p.first) {
                                t(i - b, j) = 1;
                            } else {
                                t(i - b, j) = 0;
                            }
                        }
                        tr.train(m, t);
                    }
                    if (i % 600 == 0) {
                        std::cout << "trained: " << i << std::endl;
                    }
                }
                auto et = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = et - bt;
                std::cout << "finished, used " << elapsed_seconds.count() << "sec(s)" << std::endl;
                continue;
            }
            if (s[0] == 'p') {
                s = s.substr(1);
                if (s.empty()) {
                    std::cout << "missing index" << std::endl;
                } else if (!is_digits(s)) {
                    std::cout << "invalid index: " << s << std::endl;
                } else {
                    int i = stoi(s);
                    if (i < 0 &&
                        i >= data.size()) {
                        std::cout << "invalid index: " << s << std::endl;
                    } else {
                        const auto &p = data[i];
                        std::cout << i << " target: " << int(p.first) << std::endl;
                        Matrix<float, Dynamic, 784> m = Matrix<float, Dynamic, 784>::Random(1, 784);
                        for (int i = 0; i < 784; ++i) {
                            m(0, i) = p.second[i];
                        }
                        auto pred = tr.predict(m);
                        std::cout << pred << std::endl;
                    }
                }
                continue;
            }
            if (!is_digits(s)) {
                std::cout << "unknown command: " << s << std::endl;
                std::cout << "try some number or q, quit, exit" << std::endl;
                continue;
            }
            int i = stoi(s);
            if (i >= 0 &&
                i < data.size()) {
                const auto &p = data[i];
                std::cout << i << " target: " << int(p.first) << std::endl;
                for (int l = 0; l < 28; ++l) {
                    for (int c = 0; c < 28; ++c) {
                        std::cout << to_hex(p.second[l * 28 + c]);
                    }
                    std::cout << std::endl;
                }
                continue;
            }
            std::cout << "invalid index: " << i << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "no file path" << std::endl;
        return 0;
    } else if (argc == 3) {
        std::string cs = argv[2];
        if (is_digits(cs)) {
            reserve_count = stoi(cs);
            if (reserve_count < 0 ||
                reserve_count > 10000000) {
                std::cout << "reserve count too large: " << reserve_count << std::endl;
                return 0;
            }
        }
    } else if (argc > 3) {
        std::cout << "too many params" << std::endl;
        return 0;
    }
    auto start = std::chrono::system_clock::now();
    std::vector<std::pair<unsigned char, std::vector<unsigned char>>> train_data(reserve_count);
    if (load_data(argv[1], train_data) < 0) {
        std::cout << "load train data failed" << std::endl;
        return 0;
    }
    auto finish = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = finish - start;
    std::cout << "loaded, used " << elapsed_seconds.count() << "sec(s)" << std::endl;
    std::cout << argv[1] << ": " << train_data.size() << std::endl;
    
    trainer<784, 225, 10> tr(0.3f);
    interact(train_data, tr);
    return 0;
}
