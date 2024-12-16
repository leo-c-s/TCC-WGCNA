#pragma once

#include <utility>
#include <string>

class Matrix
{
public:
    Matrix() = default;
    Matrix(int n) : Matrix(n, n) {}
    Matrix(int m, int n) : _m(m), _n(n) { _data = new float[m * n]; }
    Matrix(int m, int n, float* data) : _m(m), _n(n), _data(data) {}

    Matrix(Matrix const& other) : Matrix(other._m, other._n)
    {
        for (int i = 0; i < _m * _n; i++)
        {
            _data[i] = other._data[i];
        }
    }

    ~Matrix()
    {
        // delete[] _data;
    }

    void operator=(Matrix& other) noexcept
    {
        _m = other._m;
        _n = other._n;

        delete this->_data;

        _data = new float[_m * _n];
        for (std::size_t i = 0; i < _m * _n; i++)
        {
            _data[i] = other._data[i];
        }
    }

    float& operator()(int i, int j)
    {
        return _data[i * _n + j];
    }

    float const& operator()(int i, int j) const
    {
        return _data[i * _n + j];
    }

    bool operator==(Matrix const& other) const 
    {
        for (int i = 0; i < _m * _n; i++)
        {
            if (_data[i] != other._data[i])
            {
                return false;
            }
        }

        return true;
    }

    bool operator!=(Matrix const& other) const
    {
        for (int i = 0; i < _m * _n; i++)
        {
            if (_data[i] != other._data[i])
            {
                return true;
            }
        }

        return false;
    }

    std::pair<int, int> Size() const
    {
        return { _m, _n };
    }

    float* Data()
    {
        return _data;
    }

    float const* Data() const
    {
        return _data;
    }

    void DebugPrint() const;

private:
    int _m = 0, _n = 0;
    float* _data = nullptr;
};

Matrix read_csv(std::string filepath);
void write_csv(std::string filepath, Matrix& matrix);
