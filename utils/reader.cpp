/**
 * This code converts output of Tucker format DFT-FE from .txt to .bin for faster
 * read-in and write-out in Tucker2EI code
 *
 * @author Ian C. Lin
 *
 **/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main () {
  std::cout << "prefix: ";
  std::string prefix;
  std::cin >> prefix;

  std::cout << "file extension (enter no for files with no extention): ";
  std::string ext;
  std::cin >> ext;
  if (ext == "no") ext = "";
  else ext = "." + ext;

  int size_x, size_y, size_z;
  std::cout << "size x, y, z: ";
  std::cin >> size_x >> size_y >> size_z;

  int range0, range1;
  std::cout << "read in range (a, b): ";
  std::cin >> range0 >> range1;

  std::cout << "converting file " << prefix << std::to_string(range0) << ext << " to "  << prefix << std::to_string(range1) << ext << std::endl;

  std::cout << "size: " << size_x << ", " << size_y << ", " << size_z << std::endl;

  int size = size_x*size_y*size_z;
  for (int i = range0; i <= range1; ++i) {
    std::vector<double> v(size, 0.0);
    std::string filename = prefix + std::to_string(i) + ext;
    std::ifstream input(filename);
    std::cout << "reading in " << filename << "..." << std::endl;
    for (int iv = 0; iv < size; ++iv) {
      input >> v[iv];
    }
    input.close();
    input.clear();
   
    std::string ofilename = prefix+std::to_string(i)+".bin"; 
    std::cout << "writing out " << ofilename << "..." << std::endl;
    std::fstream fout(ofilename, std::ios::out | std::ios::binary);
    fout.write((char*)&v[0], size*sizeof(double));
    fout.close();
    fout.clear();
  }  


  return 0;
}
