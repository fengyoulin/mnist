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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

template<typename T>
class closer
{
public:
    closer(T &o) : o(o) {
    }

    ~closer() {
        o.close();
    }
    
protected:
    T &o;
};

template<int INPUT, int HIDDEN, int OUTPUT>
class trainer
{
public:
    trainer(float lrate) : inodes(INPUT), hnodes(HIDDEN), onodes(OUTPUT), lrate(lrate), ph1(nullptr), po1(nullptr) {
        pwih = new Matrix<float, INPUT, HIDDEN>();
        pwho = new Matrix<float, HIDDEN, OUTPUT>();
        *pwih = Matrix<float, INPUT, HIDDEN>::Random();
        *pwho = Matrix<float, HIDDEN, OUTPUT>::Random();
    }
    
    ~trainer() {
        delete pwho;
        delete pwih;
        if (ph1) {
            delete ph1;
        }
        if (po1) {
            delete po1;
        }
    }
    
    Matrix<float, Dynamic, HIDDEN> sigmoid(Matrix<float, Dynamic, HIDDEN> hinputs) {
        int rows = hinputs.rows();
        if (!ph1) {
            ph1 = new Array<float, Dynamic, HIDDEN>(rows, HIDDEN);
            ph1->fill(1.0f);
        } else if (ph1->rows() != rows) {
            delete ph1;
            ph1 = new Array<float, Dynamic, HIDDEN>(rows, HIDDEN);
            ph1->fill(1.0f);
        }
        return *ph1 / (*ph1 + (-hinputs.array()).exp());
    }
    
    Matrix<float, Dynamic, OUTPUT> sigmoid(Matrix<float, Dynamic, OUTPUT> oinputs) {
        int rows = oinputs.rows();
        if (!po1) {
            po1 = new Array<float, Dynamic, OUTPUT>(rows, OUTPUT);
            po1->fill(1.0f);
        } else if (po1->rows() != rows) {
            delete po1;
            po1 = new Array<float, Dynamic, OUTPUT>(rows, OUTPUT);
            po1->fill(1.0f);
        }
        return *po1 / (*po1 + (-oinputs.array()).exp());
    }
    
    void train(Matrix<float, Dynamic, INPUT> &inputs, Matrix<float, Dynamic, OUTPUT> &targets) {
        Matrix<float, Dynamic, HIDDEN> hinputs = inputs * *pwih;
        Matrix<float, Dynamic, HIDDEN> houtputs = sigmoid(hinputs);
        Matrix<float, Dynamic, OUTPUT> oinputs = houtputs * *pwho;
        Matrix<float, Dynamic, OUTPUT> outputs = sigmoid(oinputs);
        
        Matrix<float, Dynamic, OUTPUT> oerrors = targets.array() - outputs.array();
        Matrix<float, Dynamic, HIDDEN> herrors = oerrors * pwho->transpose();
        
        int orows = outputs.rows();
        if (!po1) {
            po1 = new Array<float, Dynamic, OUTPUT>(orows, OUTPUT);
            po1->fill(1.0f);
        } else if (po1->rows() != orows) {
            delete po1;
            po1 = new Array<float, Dynamic, OUTPUT>(orows, OUTPUT);
            po1->fill(1.0f);
        }
        pwho->array() += (houtputs.transpose() * (oerrors.array() * outputs.array() * (*po1 - outputs.array())).matrix()).array() * lrate;
        
        int hrows = hinputs.rows();
        if (!ph1) {
            ph1 = new Array<float, Dynamic, HIDDEN>(hrows, HIDDEN);
            ph1->fill(1.0f);
        } else if (ph1->rows() != hrows) {
            delete ph1;
            ph1 = new Array<float, Dynamic, HIDDEN>(hrows, HIDDEN);
            ph1->fill(1.0f);
        }
        pwih->array() += (inputs.transpose() * (herrors.array() * houtputs.array() * (*ph1 - houtputs.array())).matrix()).array() * lrate;
    }
    
    Matrix<float, Dynamic, OUTPUT> predict(Matrix<float, Dynamic, INPUT> &inputs) {
        Matrix<float, Dynamic, HIDDEN> hinputs = inputs * *pwih;
        Matrix<float, Dynamic, HIDDEN> houtputs = sigmoid(hinputs);
        Matrix<float, Dynamic, OUTPUT> oinputs = houtputs * *pwho;
        Matrix<float, Dynamic, OUTPUT> outputs = sigmoid(oinputs);
        return outputs;
    }
    
    Matrix<float, INPUT, HIDDEN> getIh() {
        return *pwih;
    }
    
    Matrix<float, HIDDEN, OUTPUT> getHo() {
        return *pwho;
    }
    
    int saveModel(const char *path) {
        std::ofstream os(path, std::ios::out);
        if (!os.is_open()) {
            std::cout << "cannot open: " << path << std::endl;
            return -1;
        }
        closer<std::ofstream> co(os);
        int rows = pwih->rows();
        int cols = pwih->cols();
        os << rows << ',' << cols << '\n';
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (c > 0) {
                    os << ',';
                }
                os << (*pwih)(r, c);
            }
            os << '\n';
        }
        rows = pwho->rows();
        cols = pwho->cols();
        os << rows << ',' << cols << '\n';
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (c > 0) {
                    os << ',';
                }
                os << (*pwho)(r, c);
            }
            os << '\n';
        }
        return 0;
    }
    
    int loadModel(const char *path) {
        std::ifstream is(path, std::ios::in);
        if (!is.is_open()) {
            std::cout << "cannot open: " << path << std::endl;
            return -1;
        }
        closer<std::ifstream> ci(is);
        int rows = 0, cols = 0;
        std::string line;
        if (std::getline(is, line)) {
            if (parse_dim(line, rows, cols) < 0 || rows != pwih->rows() || cols != pwih->cols()) {
                return -1;
            }
        }
        int i = 0;
        while (i < rows && std::getline(is, line)) {
            std::vector<float> rd = parse_row(line);
            if (rd.size() != cols) {
                return -1;
            }
            for (int j = 0; j < cols; ++j) {
                (*pwih)(i, j) = rd[j];
            }
            ++i;
        }
        if (i != rows) {
            return -1;
        }
        if (std::getline(is, line)) {
            if (parse_dim(line, rows, cols) < 0 || rows != pwho->rows() || cols != pwho->cols()) {
                return -1;
            }
        }
        i = 0;
        while (i < rows && std::getline(is, line)) {
            std::vector<float> rd = parse_row(line);
            if (rd.size() != cols) {
                return -1;
            }
            for (int j = 0; j < cols; ++j) {
                (*pwho)(i, j) = rd[j];
            }
            ++i;
        }
        if (i != rows) {
            return -1;
        }
        return 0;
    }
    
protected:
    std::vector<float> parse_row(const std::string &s) {
        std::vector<float> v;
        std::string elem;
        const char *b = s.c_str(), *c = b, *e = b + s.length();
        while (c <= e) {
            if (c == e || *c == ',') {
                elem.assign(b, c - b);
                v.push_back(stof(elem));
                b = c + 1;
            }
            ++c;
        }
        return v;
    }
    
    int parse_dim(const std::string &s, int &rows, int &cols) {
        int ret = -1;
        std::string str;
        const char *b = s.c_str(), *c = b, *e = b + s.length();
        while (c <= e) {
            if (*c == ',') {
                str.assign(b, c - b);
                rows = stoi(str);
                str.assign(c + 1, e - c - 1);
                cols = stoi(str);
                ret = 0;
                break;
            }
            ++c;
        }
        return ret;
    }
    
    const int inodes;
    const int hnodes;
    const int onodes;
    
    float lrate;
    
    Matrix<float, INPUT, HIDDEN> *pwih;
    Matrix<float, HIDDEN, OUTPUT> *pwho;
    
    Array<float, Dynamic, HIDDEN> *ph1;
    Array<float, Dynamic, OUTPUT> *po1;
};
