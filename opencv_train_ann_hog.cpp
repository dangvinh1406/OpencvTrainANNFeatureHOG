#include <iostream>
#include <vector>
#include "objdetect.hpp"
#include "ml.hpp"

using namespace std;

int main(int argc, const char** argv){
	// declare input variables;
	string positive_path, negative_path;
	string saving_file_name;
	int width, height;
	vector<cv::String> positive_image_names, negative_image_names, temporary_names;

	cout << "TRAINING ARTIFICIAL NEURAL NETWORK USING HOG FEATURE" << endl;
	if (argc <= 5){
		cout << "Enter directory of the folder contains positive images: (i.e /home/images/) " << endl;
		cin >> positive_path;
		
		cout << "Enter directory of the folder contains negative images: (i.e /home/images/)" << endl;
		cin >> negative_path;

		cout << "Enter width of image: " << endl;
		cin >> width;
		cout << "Enter height of image: " << endl;
		cin >> height;

		cout << "Enter the filename to save the model:" << endl;
		cin >> saving_file_name;
	}
	else {
		positive_path = string(argv[1]);
		negative_path = string(argv[2]);
		width = atoi(argv[3]);
		height = atoi(argv[4]);
		saving_file_name = string(argv[5]);
	}

	cout << "Collecting images....." << endl;
	temporary_names.clear();
	glob(positive_path+"*.png", temporary_names, false);
	positive_image_names.insert(positive_image_names.end(), temporary_names.begin(), temporary_names.end());
	temporary_names.clear();
	glob(positive_path+"*.jpg", temporary_names, false);
	positive_image_names.insert(positive_image_names.end(), temporary_names.begin(), temporary_names.end());
	temporary_names.clear();
	glob(positive_path+"*.pgm", temporary_names, false);
	positive_image_names.insert(positive_image_names.end(), temporary_names.begin(), temporary_names.end());
	temporary_names.clear();
	cout << positive_image_names.size() << " positive images collected....." << endl;

	cout << "Collecting images....." << endl;
	temporary_names.clear();
	glob(negative_path+"*.png", temporary_names, false);
	negative_image_names.insert(negative_image_names.end(), temporary_names.begin(), temporary_names.end());
	temporary_names.clear();
	glob(negative_path+"*.jpg", temporary_names, false);
	negative_image_names.insert(negative_image_names.end(), temporary_names.begin(), temporary_names.end());
	temporary_names.clear();
	glob(negative_path+"*.pgm", temporary_names, false);
	negative_image_names.insert(negative_image_names.end(), temporary_names.begin(), temporary_names.end());
	temporary_names.clear();
	cout << negative_image_names.size() << " negative images collected....." << endl;
	
	// create empty matrices for storing data
	Mat training_data; // size: number_of_training_samples x attributes_per_sample
	Mat training_classification; // size: number_of_training_samples x number_of_classes

	// create HOG object
	HOGDescriptor HOG;
	cout << "Computing HOG feature of positive images....." << endl;
	for (size_t i = 0; i < positive_image_names.size(); i++){
		Mat image = imread(positive_image_names[i]);
		if (image.size().width < width || image.size().height < height){
			resize(image, image, Size(width, height));
		}
		else{
			int dw = image.size().width - width;
			int dh = image.size().height - height;
			Rect r(dw/2, dh/2, width, height);
			Mat temp_image = image(r);
			resize(temp_image, image, Size(width, height));
		}
		vector<float> descriptor;
		HOG.compute(image, descriptor);
		Mat feature(descriptor, true);
		Mat_<float> label = (Mat_<float>(1,2)<< 1, 0);
		if (training_data.empty()){
			training_data = feature.clone();
			training_classification = label.clone();
		}
		else{
			hconcat(training_data, feature, training_data);
			vconcat(training_classification, label, training_classification);
		}
		cout << positive_image_names[i] << "\r";
	}
	cout << endl << "Positive images computed....." << endl;

	cout << "Sliding and computing HOG feature of negative images....." << endl;
	for (size_t i = 0; i < negative_image_names.size(); i++){
		Mat image = imread(negative_image_names[i]);
		if (image.size().width >= width && image.size().height >= height){
			for (int x = 0; x < image.size().width-width; x += width){
				for (int y = 0; y < image.size().height-height; y += height){
					Mat subimage = image(Rect(x,y,width,height));
					vector<float> descriptor;
					HOG.compute(subimage, descriptor);
					Mat feature(descriptor, true);
					Mat_<float> label = (Mat_<float>(1,2)<< 0, 1);
					if (training_data.empty()){
						training_data = feature.clone();
						training_classification = label.clone();
					}
					else{
						hconcat(training_data, feature, training_data);
						vconcat(training_classification, label, training_classification);
					}
				}
			}
			cout << negative_image_names[i] << "\r";
		}
	}
	transpose(training_classification, training_classification);
	training_data.convertTo(training_data, CV_32FC1);
	training_classification.convertTo(training_classification, CV_32FC1);
	cout << endl << "Negative images computed....." << endl;
	
	// set parameters of dataset
	int number_of_training_samples = training_data.cols;
	int number_of_classes = 2; // true or false
	int attributes_per_sample = training_data.rows;

	cout << "Training feature space size: " << training_data.size() << endl;
	cout << "Labeling space size: " << training_classification.size() << endl;

	// create a matrix indicates structure of layers	
	Mat_<int> structure_of_network = (Mat_<int>(1,3)<<
		(int)attributes_per_sample, 10, (int)number_of_classes);// 3 layers: input -> hidden layer 1 (10 node) -> output

	// create an empty ANN object
	Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();

	// set structure of layers in network
	ann->setLayerSizes(structure_of_network);

	// set activation function in network: IDENTITY, SIGMOID_SYM or GAUSSIAN
	ann->setActivationFunction(cv::ml::ANN_MLP::IDENTITY);

	// training
	cout << "Training....." << endl;
	ann->train(training_data, cv::ml::COL_SAMPLE, training_classification);
	cout << "Training complete." << endl;

	// saving
	ann->save(saving_file_name);
	return 0;
}