#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;

int start = 0;

int main(int argc, char *argv[]) {
	int N;
	fstream f1, f2;
	f1.open(argv[1], ios_base::in);
	f2.open(argv[2], ios_base::in);
	f1.read((char *)&N, sizeof(N));
	f2.read((char *)&N, sizeof(N));
	float mean = 0.0;
	int count = 0;
	for (int i = 0; i < N; i++) {
		if (i < start) {
			f1.seekg(sizeof(float), ios_base::cur);
			continue;
		}
		else {
			float temp;
			f1.read((char *)&temp, sizeof(temp));
			mean += temp;
			count += 1;
		}
	}

	for (int i = 0; i < N; i++) {
		if (i < start) {
			f2.seekg(sizeof(float), ios_base::cur);
			continue;
		}
		else {
			float temp;
			f2.read((char *)&temp, sizeof(temp));
			mean += temp;
			count += 1;
		}
	}

	mean /= count;

	float std_dev = 0.0;
	f1.seekg(sizeof(int), ios_base::beg);
	f2.seekg(sizeof(int), ios_base::beg);
	for (int i = 0; i < N; i++) {
		if (i < start) {
			f1.seekg(sizeof(float), ios_base::cur);
			continue;
		}
		else {
			float temp;
			f1.read((char *)&temp, sizeof(temp));
			std_dev += pow((temp - mean), 2);
		}
	}

	for (int i = 0; i < N; i++) {
		if (i < start) {
			f2.seekg(sizeof(float), ios_base::cur);
			continue;
		}
		else {
			float temp;
			f2.read((char *)&temp, sizeof(temp));
			std_dev += pow((temp - mean), 2);
		}
	}
	std_dev /= count;

	std_dev = pow(std_dev, 0.5);

	cout << "mean: " << mean << endl;
	cout << "std_dev: " << std_dev << endl;


}