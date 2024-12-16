#include "Matrix.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cstdlib>

void Matrix::DebugPrint() const
{
    for (int i = 0; i < _m; i++)
    {
        for (int j = 0; j < _n; j++)
        {
            std::cout << _data[i * _n + j] << ' ';
        }
        std::cout << '\n';
    }
}

static std::vector<std::string> split_string(std::string str, char c)
{
    std::vector<std::string> items;

    std::size_t i = 0;
    std::size_t j = str.find(c);

    while (i < str.length() && j != std::string::npos)
    {
        items.push_back(str.substr(i, j - i));
        i = j + 1;
        j = str.find(c, i);
    }

    if (i < str.length() && j == std::string::npos)
    {
        items.push_back(str.substr(i));
    }

    return items;
}

Matrix read_csv(std::string filepath)
{
    // std::cout << "reading csv\n";
    std::ifstream input(filepath, std::ios::in | std::ios::binary);

    std::vector<std::vector<float>> rows;

    std::string line;
    int i = 0;
    while (std::getline(input, line))
    {
        i++;
        // std::cout << "read line: " << line << '\n';
        std::vector<float> row;
        auto items = split_string(line, ',');

        int j = 0;
        for (auto const& item : items)
        {
            j++;
            try {
                row.push_back(std::stod(item));
            }
            catch (std::exception e) {
                std::cerr << "Failed to read item " << i << ", " << j
                    << ": (" << item << ") "
                    << e.what() << "\n";
                throw e;
            }
        }

        rows.push_back(row);
    }

    if (rows.size() == 0 || rows[0].size() == 0)
    {
        return {};
    }
    std::cout << "size of matrix: " << rows.size() * rows[0].size() << '\n';
    int const m = rows.size();
    int const n = rows[0].size();
    float* data = new float[m * n];

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            data[i * n + j] = rows[i][j];
        }
    }

    return Matrix(rows.size(), rows[0].size(), data);
}

void write_csv(std::string filepath, Matrix& matrix)
{
    int m, n;
    {
        auto size = matrix.Size();
        m = size.first;
        n = size.second;
    }

    std::ofstream outfile(filepath, std::ios::binary);
    std::cout << "writing to " << filepath << "\n";
    for (int i = 0; i < m; i++)
    {
        outfile << matrix(i, 0);
        for (int j = 1; j < n; j++)
        {
            outfile << "," << matrix(i, j);
        }
        outfile << '\n';
    }
    outfile.flush();
}
