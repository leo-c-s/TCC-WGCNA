#include "Tree.hpp"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <thread>

Tree::Tree(int value) : m_left(nullptr), m_right(nullptr), m_dist(-1.0f), m_value(value), m_size(1)
{}

Tree::Tree(float dist, Tree* left, Tree* right)
: m_left(left), m_right(right), m_dist(dist), m_size(0), m_value(cuda::std::nullopt)
{
	m_size = left->m_size + right->m_size;
}

std::map<int, int> Tree::DynamicCut(float baseHeight, int minClusterSize) const
{
	int n = m_size;

	std::vector<float> heights;

	getInOrderDistances(heights);

	auto clusters = cut(heights, baseHeight, minClusterSize);
	bool hasNewClusters = !clusters.empty();

	clusters.insert(clusters.begin(), 0);
	clusters.push_back(heights.size());

	while(hasNewClusters)
	{
		std::vector<int> newClusters;

		hasNewClusters = false;

		std::vector<std::thread> threads;
		std::vector<std::vector<int>> threadResults(clusters.size());

		for (int i = 1; i < clusters.size(); i++)
		{
			std::vector<float> currentCluster(heights.begin() + clusters[i - 1], heights.begin() + clusters[i]);
			threads.push_back(std::thread([=, &threadResults]() {
				threadResults[i - 1] = adaptiveCut(currentCluster, minClusterSize);
			}));
		}

		for (int i = 1; i < clusters.size(); i++)
		{
			newClusters.push_back(clusters[i - 1]);
			
			threads[i - 1].join();
			auto currentClusters = threadResults[i - 1];

			if (currentClusters.size() > 0 and currentClusters[0] >= minClusterSize)
			{
				hasNewClusters = true;
				newClusters.push_back(currentClusters[0] + clusters[i - 1]);
			}

			for (int j = 1; j < currentClusters.size(); j++)
			{
				if (currentClusters[j] - currentClusters[j - 1] >= minClusterSize)
				{
					hasNewClusters = true;
					newClusters.push_back(currentClusters[j] + clusters[i - 1]);
				}
			}
		}

		if (hasNewClusters)
		{
			clusters = newClusters;
			clusters.push_back(heights.size());
		}
	}

	std::map<int, int> nodeToCluster{};

	std::vector<int> values;
	values.reserve(m_size);

	getLeafValues(values);

	for (std::size_t i = 1; i < clusters.size(); i++)
	{
		auto start = values.begin() + clusters[i - 1];
		auto end = values.begin() + clusters[i];

		std::transform(start, end, std::inserter(nodeToCluster, nodeToCluster.end()), [i](auto leafValue) {
			return std::pair<int,int> { leafValue, i };
		});
	}

	return nodeToCluster;
}

void Tree::getInOrderDistances(std::vector<float>& dists) const
{
	if (IsLeaf())
	{
		return;
	}

	m_left->getInOrderDistances(dists);

	dists.push_back(m_dist);

	m_right->getInOrderDistances(dists);
}

void Tree::getLeafValues(std::vector<int>& values) const
{
	if (IsLeaf())
	{
		values.push_back(Value());
		return;
	}

	m_left->getLeafValues(values);
	m_right->getLeafValues(values);
}

std::vector<int> Tree::cut(std::vector<float> const& heights, float level, int minClusterSize)
{
	if (heights.empty()) return {};

	std::vector<float> heightDiffs{};
	heightDiffs.reserve(heights.size());
	std::transform(heights.begin(), heights.end(), std::back_inserter(heightDiffs), [=](float h) { return h - level; });

	std::vector<std::size_t> transitionPoints{};
	transitionPoints.reserve(heights.size());
	for (std::size_t i = 0; i < heightDiffs.size() - 1; i++)
	{
		if (heightDiffs[i] > 0.0f and heightDiffs[i + 1] <= 0.0f)
		{
			transitionPoints.push_back(i);
		}
	}

	std::vector<int> breakpoints{};
	breakpoints.reserve(heights.size());

	for (std::size_t i : transitionPoints) {
		if (i == 0)
		{
			continue;
		}

		for (std::size_t j = i - 1; j > 0 and heightDiffs[j] >= 0.0f; j--)
		{
			if (heightDiffs[j] > 0.0f and heightDiffs[j - 1] < 0.0f) // and i - j >= minClusterSize)
			{
				breakpoints.push_back(j);
			}
		}
	}

	return breakpoints;
}

std::vector<int> Tree::adaptiveCut(std::vector<float> heights, int minClusterSize)
{
	double sum = std::accumulate(heights.begin(), heights.end(), 0.0);

	float level = sum / heights.size();
	auto clusters = cut(heights, level, minClusterSize);

	if (clusters.size() == 0)
	{
		float level_d = 0.5f * level + 0.5f * *std::min_element(heights.begin(), heights.end());
		clusters = cut(heights, level_d, minClusterSize);

		if (clusters.size() == 0)
		{
			float level_u = 0.5f * level + 0.5f * *std::max_element(heights.begin(), heights.end());
			clusters = cut(heights, level_u, minClusterSize);
		}
	}

	return clusters;
}
