#!/bin/bash
#
#
#  An installer for the triangle 2D mesh generator
#
#  Written by Jonathan Roveto  - veto1024@gmail.com
#
#################################################################3

echo "Cloning from Triangle GitHub"
git clone https://github.com/libigl/triangle.git
cd triangle
mkdir bin
echo "Compiling Triangle"
gcc -O -o bin/triangle triangle.c -lm
make
cd bin
chmod +x triangle
cd ..
cd ..
mkdir triangle_bin
mv triangle/bin/triangle triangle_bin/triangle
rm -Rf triangle
echo ""
echo "The triangle binary is now in triangle_bin. This needs to be linked to a folder on your \$PATH"
echo ""
echo "Use"
echo "\$ echo \$PATH"
echo "to find your \$PATH and pick a suitable directory. If /usr/bin is on your path, for example,"
echo "Use "
echo "\$ sudo ln -s ${PWD}/triangle_bin/triangle /usr/bin/triangle"
echo "to link it! This will allow you to perform the command"
echo "\$ triangle"
echo "which is needed by neutpy to run the Triangle meshing routine."
exit 0