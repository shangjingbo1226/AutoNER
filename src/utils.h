#ifndef __UTILS_H__
#define __UTILS_H__

#include "omp.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
using namespace std;

namespace Utils
{

const double EPS = 1e-8;

double sqr(double x)
{
    return x * x;
}

int sign(double x)
{
    return x < -EPS ? -1 : x > EPS;
}

void myAssert(bool flg, string msg)
{
    if (!flg) {
        cerr << "[ERROR] " << msg << endl;
        exit(-1);
    }
}

FILE* tryOpen(string filename, string param = "r")
{
    FILE* ret = fopen(filename.c_str(), param.c_str());
    if (ret == NULL) {
        fprintf(stderr, "Error while opening %s under parameter %s.\n", filename.c_str(), param.c_str());
    }
    return ret;
}

const int MAX_LENGTH = 100000000;

char line[MAX_LENGTH + 1];

inline bool getLine(FILE* in)
{
    bool hasNext = fgets(line, MAX_LENGTH, in);
    int length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
        -- length;
    }
    line[length] = 0;
    return hasNext;
}


inline vector<string> splitBy(const string &line, char sep)
{
    vector<string> tokens;
    string token = "";
    for (size_t i = 0; i < line.size(); ++ i) {
        if (line[i] == sep) {
            if (token != "") {
                tokens.push_back(token);
            }
            token = "";
        } else {
            token += line[i];
        }
    }
    if (token != "") {
        tokens.push_back(token);
    }
    return tokens;
}

string normalize(string s)
{
    /*int l = s.find('(');
    int r = s.find(')');
    if (l != -1 && r != -1 && l < r) {
        s = s.substr(0, l - 1) + s.substr(r + 1);
    }*/
    int l = 0, r = (int)s.size() - 1;
    while (l < r && isspace(s[l])) {
        ++ l;
    }
    while (l < r && isspace(s[r])) {
        -- r;
    }
    return s.substr(l, r - l + 1);
}

inline string strip(const string& s)
{
    int l = 0, r = (int)s.size() - 1;
    while (l < r && isspace(s[l])) {
        ++ l;
    }
    while (l < r && isspace(s[r])) {
        -- r;
    }
    return s.substr(l, r - l + 1);
}

vector<string> simpleJsonToList(string s)
{
    vector<string> ret;
    bool inside = false;
    string token = "";
    for (int i = 0; i < s.size(); ++ i) {
        if (s[i] == '"') {
            inside ^= 1;
            if (inside == false) {
                ret.push_back(token);
                token = "";
            }
        } else {
            if (inside) {
                token += s[i];
            }
        }
    }
    assert(inside == false);
    return ret;
}

string toUpper(const string& s)
{
    string ret = "";
    for (int i = 0; i < s.size(); ++ i) {
        ret += toupper(s[i]);
    }
    return ret;
}

string toLower(const string& s)
{
    string ret = "";
    for (int i = 0; i < s.size(); ++ i) {
        ret += tolower(s[i]);
    }
    return ret;
}

const string SEPARATORS = "/.,-()!?~@#$%^&*[]\n";

bool isSeparator(char ch)
{
    if (isspace(ch)) {
        return true;
    }
    if (SEPARATORS.find(ch) != -1) {
        return true;
    }
    return false;
}

bool isSeparator(const string& token)
{
    return token.size() == 1 && isSeparator(token[0]);
}

inline bool isRealSeparator(const string& token)
{
    return token == "\n";
    return isSeparator(token) && SEPARATORS.find(token[0]) != -1;
}

bool isUpper(const string& token)
{
    for (char ch : token) {
        if (!isupper(ch)) {
            return false;
        }
    }
    return true;
}

vector<string> simpleTokenize(const string& s)
{
    vector<string> ret;
    string token = "";
    for (char ch : s) {
        if (isSeparator(ch)) {
            if (token != "") {
                ret.push_back(token);
                token = "";
            }
            string temp = "";
            temp += ch;
            ret.push_back(temp);
        } else {
            token += ch;
        }
    }
    if (token != "") {
        ret.push_back(token);
    }

    int total = 0;
    for (const string& token : ret) {
        total += token.size();
    }
    if (total != s.size()) {
        cerr << "ERROR in simpleTokenize! " << total << endl;
        cerr << "=== Raw === " << s.size() << endl << s << endl;
        cerr << "=== Tokenized === " << ret.size() << endl;
        for (const string& token : ret) {
            cerr << token << endl;
        }
        assert(total == s.size());
    }

    return ret;
}

vector<string> simpleTokenizeNonSep(const string& s, bool keepCapital = false)
{
    vector<string> tokens = simpleTokenize(s), ret;
    for (const string& token : tokens) {
        if (isSeparator(token) && !isRealSeparator(token)) {
            continue;
        }
        if (keepCapital) {
            ret.push_back(token);
        } else {
            ret.push_back(toLower(token));
        }
    }
    return ret;
}

string generateSignature(string s)
{
    stringstream in(s);
    string ret = "";
    for (string token; in >> token;) {
        ret += tolower(token[0]);
    }
    return ret;
}

}


namespace RandomNumbers
{
struct RandomNumberGenerator
{
    unsigned int MT[624];
    int index;

    void init(int seed = 1) {
        MT[0] = seed;
        for (int i = 1; i < 624; ++ i) {
            MT[i] = (1812433253UL * (MT[i-1] ^ (MT[i-1] >> 30)) + i);
        }
        index = 0;
    }

    void generate() {
        const unsigned int MULT[] = {0, 2567483615UL};
        for (int i = 0; i < 227; ++ i) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i+397] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        for (int i = 227; i < 623; ++ i) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i-227] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        unsigned int y = (MT[623] & 0x8000000UL) + (MT[0] & 0x7FFFFFFFUL);
        MT[623] = MT[623-227] ^ (y >> 1);
        MT[623] ^= MULT[y&1];
    }

    unsigned int rand() {
        if (index == 0) {
            generate();
        }

        unsigned int y = MT[index];
        y ^= y >> 11;
        y ^= y << 7  & 2636928640UL;
        y ^= y << 15 & 4022730752UL;
        y ^= y >> 18;
        index = index == 623 ? 0 : index + 1;
        return y;
    }

    int next(int x) { // [0, x)
        return rand() % x;
    }

    int next(int a, int b) { // [a, b)
        return a + (rand() % (b - a));
    }

    double nextDouble() { // (0, 1)
        return (rand() + 0.5) * (1.0 / 4294967296.0);
    }
};

static vector<RandomNumberGenerator> rng;

void initialize()
{
    int nthread = omp_get_max_threads();
//cerr << "# of threads = " << nthread << endl;
    rng.resize(nthread);
    RandomNumberGenerator seeds;
    seeds.init(19910724);
    for (int i = 0; i < nthread; ++ i) {
        rng[i].init(seeds.rand());
    }
}

}

#endif