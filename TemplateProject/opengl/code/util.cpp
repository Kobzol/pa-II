#include "util.h"

#include <fstream>
#include <sstream>

std::string loadFile(const std::string& path)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw "File not found: " + path;
	}

	std::stringstream buffer;
	buffer << file.rdbuf();

	return buffer.str();
}
