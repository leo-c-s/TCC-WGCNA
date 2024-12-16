#pragma once

#include <cuda/std/optional>
#include <vector>
#include <map>
#include <functional>

class Tree
{
public:
	explicit Tree(int value);
	Tree(float dist, Tree* left, Tree* right);

	inline bool IsLeaf() const
	{
		return m_value.has_value();
	}

	inline int Value() const
	{
		return m_value.value();
	}

	inline float Dist() const
	{
		return m_dist;
	}

	std::map<int, int> DynamicCut(float baseHeight, int minClusterSize=30) const;

private:
	float m_dist;
	Tree* m_left;
	Tree* m_right;
	cuda::std::optional<int> m_value;
	int m_size;

	void getInOrderDistances(std::vector<float>& dists) const;
	void getLeafValues(std::vector<int>& dists) const;

	static std::vector<int> cut(std::vector<float> const& heights, float level, int minClusterSize);
	static std::vector<int> adaptiveCut(std::vector<float> heights, int minClusterSize);
};
