﻿#ifndef UMML_HEARTDISEASE_INCLUDED
#define UMML_HEARTDISEASE_INCLUDED

/*
 Machine Learning Artificial Intelligence Library

 FILE:   heartdisease.hpp
 AUTHOR: Anastasios Milikas (amilikas@csd.auth.gr)
 
 Namespace
 ~~~~~~~~~
 umml
 
 Requirements
 ~~~~~~~~~~~~
 umml vector
 umml matrix
  
 Description
 ~~~~~~~~~~~
 Heart Disease Health Indicators Dataset
 https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset
  
 Features (columns):
 Classification, HighBP, HighChol, CholCheck, Smoker, Stroke, Diabetes, PhysActivity,
 Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex, Income 
 
 Class: 
 0 -- No heart disease (857)
 1 -- heart disease (143)
  
 Total samples:  1000
 Missing values: 0
*/

#include "../umat.hpp"


namespace umml {

	
/// load_heartdisease
/// loads the heart disease data in matrix 'X' and the labels in vector 'y'
template <typename XT, typename YT>
void load_heartdisease(umat<XT>& X, uvec<YT>& y)
{
	assert(X.dev()==device::CPU && y.dev()==device::CPU);
	
	std::vector<std::vector<XT>> hd_data = {
	0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,3,
	1,1,1,1,1,0,2,0,0,0,0,1,0,0,1,6,
	0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,
	0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	1,1,1,1,1,1,2,0,1,1,0,1,0,1,0,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,7,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,4,
	1,1,1,1,1,0,2,0,1,1,0,1,0,1,0,1,
	0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,3,
	0,0,0,1,1,0,2,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,
	0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,7,
	0,1,1,1,0,0,2,0,0,1,0,1,0,1,0,6,
	1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,3,
	0,0,1,1,1,1,0,1,0,1,0,1,1,0,0,2,
	0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,3,
	0,0,0,1,1,0,2,1,0,0,0,1,0,0,1,6,
	0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,7,
	0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,8,
	1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,4,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,3,
	0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,5,
	0,1,0,1,0,0,2,1,1,1,0,1,0,0,0,4,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	1,1,1,1,1,1,2,0,0,1,0,1,0,1,1,5,
	1,1,1,1,1,0,2,0,0,1,0,1,0,0,1,4,
	0,1,1,1,1,0,2,0,1,1,0,1,0,1,0,7,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,1,2,1,0,0,0,1,0,1,0,4,
	0,1,0,1,1,0,0,1,1,1,0,0,0,1,1,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,2,
	0,1,1,1,1,0,2,0,0,0,0,1,0,0,0,3,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,0,1,1,0,0,1,1,1,0,0,0,0,1,3,
	0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,1,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,7,
	0,1,0,1,0,0,2,0,1,0,0,1,0,0,0,4,
	0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,3,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,0,8,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,8,
	0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,3,
	0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,2,
	1,1,1,1,1,0,0,0,1,1,0,1,0,0,1,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,1,6,
	0,1,1,1,1,0,2,0,0,1,0,1,0,1,0,2,
	1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,1,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,3,
	0,1,1,1,1,0,2,0,0,1,0,1,0,1,0,5,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,8,
	0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,8,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	1,1,1,1,1,0,0,1,0,1,0,1,0,0,1,6,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,5,
	0,1,0,1,0,0,2,1,1,1,0,1,0,0,0,4,
	0,1,0,1,1,0,0,0,0,0,0,1,0,1,0,2,
	0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,5,
	0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,3,
	1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,6,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,4,
	0,1,0,1,1,0,2,1,1,1,0,1,0,1,0,2,
	1,1,1,1,1,1,2,0,1,0,0,1,1,0,0,3,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,1,5,
	1,1,1,1,0,0,2,1,0,0,0,1,0,0,1,8,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,
	0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,7,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,4,
	0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,6,
	0,1,0,1,0,0,2,1,0,1,0,1,0,0,1,8,
	1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,2,
	0,1,1,1,1,0,0,1,0,1,1,1,0,0,1,3,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,8,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,8,
	0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,7,
	0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,4,
	1,0,1,1,0,0,2,1,0,0,0,1,0,1,0,2,
	0,0,0,1,1,0,0,1,0,1,1,1,0,0,0,7,
	0,1,1,1,0,0,2,0,0,1,0,1,0,1,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,0,0,1,0,0,1,0,0,1,8,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	1,1,1,1,1,0,2,1,0,1,0,1,0,0,1,6,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,1,1,1,1,0,2,1,1,1,1,1,0,0,1,7,
	0,0,1,1,1,1,2,1,0,0,0,1,1,1,0,2,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,7,
	1,1,1,1,1,0,2,0,1,1,0,1,0,1,0,3,
	0,0,0,1,0,0,2,1,0,1,0,0,1,0,0,3,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,3,
	1,1,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,0,8,
	1,0,1,1,1,0,0,1,1,1,0,1,0,0,1,5,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,1,1,1,0,0,2,1,1,1,0,1,0,0,1,7,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,2,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,2,
	0,1,1,1,1,0,2,1,0,1,0,1,0,0,1,5,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,1,1,1,0,0,0,1,0,1,0,1,0,1,0,3,
	0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,6,
	0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,4,
	0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,6,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,6,
	0,0,0,1,1,0,0,1,1,1,0,1,1,1,1,6,
	1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,7,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,7,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,7,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,6,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,2,0,1,1,0,1,0,0,0,7,
	1,0,1,1,0,0,0,0,1,0,0,1,0,1,0,4,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,7,
	0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,7,
	1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,5,
	0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,4,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,8,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,1,3,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,1,0,1,0,1,0,0,0,6,
	1,1,1,1,1,0,0,0,1,1,0,1,1,1,0,2,
	0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,4,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,5,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,1,0,2,1,0,1,0,1,0,0,1,6,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,0,0,1,0,1,1,1,0,7,
	0,0,0,1,1,0,0,0,0,1,0,0,1,0,1,3,
	0,1,1,1,1,1,0,0,0,1,0,0,1,1,0,2,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,2,0,0,0,0,1,0,0,0,5,
	0,1,1,1,1,0,2,1,0,1,1,1,0,0,1,8,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,1,
	0,1,1,1,1,1,0,1,0,0,0,0,0,1,1,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,1,0,0,0,1,0,1,1,0,0,0,4,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,1,1,1,0,0,2,0,0,1,0,0,0,0,0,7,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,6,
	0,1,0,1,0,0,2,1,1,1,0,1,0,0,0,4,
	0,1,0,1,0,0,2,1,0,1,0,1,1,1,0,6,
	0,1,1,1,1,0,0,0,0,0,0,1,0,1,0,6,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,8,
	0,1,1,1,1,0,2,0,1,0,0,1,0,1,0,5,
	1,0,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,2,1,0,1,0,1,0,0,0,5,
	0,1,0,1,1,0,0,0,1,1,0,1,0,0,0,4,
	0,0,1,1,1,0,2,1,0,0,0,1,0,0,1,5,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,0,0,1,0,1,0,1,1,8,
	0,1,0,1,0,0,2,0,0,0,0,1,0,0,0,8,
	0,0,0,0,1,0,0,1,1,1,0,1,0,0,1,4,
	0,1,0,1,0,0,2,0,1,0,0,1,0,1,0,6,
	0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,4,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,7,
	0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,7,
	0,1,1,1,1,0,0,1,0,0,0,1,1,1,1,6,
	0,1,1,1,1,0,2,1,1,1,0,1,0,0,1,6,
	1,1,1,1,0,0,2,0,1,1,0,1,0,1,0,3,
	0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,8,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,5,
	1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,2,
	0,0,0,1,1,0,0,0,0,1,0,0,1,0,1,5,
	0,1,0,1,1,0,2,1,0,1,0,1,0,1,1,8,
	1,1,1,1,0,0,2,1,1,1,0,1,0,1,0,7,
	1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,6,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,8,
	0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,3,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,5,
	0,1,1,1,1,0,0,0,1,1,1,1,0,0,1,6,
	1,1,1,1,1,0,2,1,0,1,0,1,0,1,1,8,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,0,1,1,0,1,0,1,1,6,
	0,1,0,1,1,0,0,0,0,1,0,1,0,1,0,2,
	0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,8,
	0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,5,
	1,0,0,1,1,0,0,0,1,1,0,1,0,0,1,7,
	0,1,0,1,1,0,0,0,0,0,0,1,0,0,1,5,
	0,1,1,1,0,0,1,0,0,1,0,1,0,0,0,8,
	0,1,0,1,0,0,2,1,1,1,0,1,0,0,0,5,
	0,1,1,1,1,0,0,0,1,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,7,
	1,1,1,1,1,1,2,0,1,1,0,1,0,0,0,4,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,8,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,2,0,1,1,0,1,0,0,0,2,
	0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,8,
	0,1,1,1,1,1,0,0,1,1,0,1,0,0,0,6,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,4,
	0,0,0,1,1,0,0,1,1,0,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,8,
	0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,3,
	0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,8,
	0,1,1,1,1,0,2,1,1,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,0,1,1,0,1,1,1,0,7,
	1,1,1,1,1,0,2,0,1,1,0,1,1,1,0,2,
	0,0,1,1,1,0,0,1,1,1,0,1,1,0,1,6,
	0,1,0,1,1,0,0,1,0,1,0,0,1,0,0,1,
	0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,4,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,7,
	0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,7,
	0,1,1,1,1,1,2,0,0,1,0,1,0,0,0,1,
	0,1,1,1,1,0,0,1,1,0,0,1,0,1,1,1,
	0,1,0,1,0,0,2,1,1,0,0,1,0,0,0,4,
	0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,5,
	1,1,1,1,1,0,0,1,0,1,0,1,0,1,0,7,
	0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,6,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,1,0,0,1,0,1,0,1,1,0,1,4,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,4,
	0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,2,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,6,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,1,8,
	0,1,0,1,1,0,2,0,0,0,0,1,0,1,1,6,
	0,1,1,1,1,0,1,0,1,1,0,1,0,1,0,6,
	0,1,1,1,1,0,2,1,1,1,0,1,0,0,1,7,
	0,1,1,1,0,0,2,1,0,0,0,1,0,1,1,8,
	0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,8,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,8,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,6,
	0,0,0,1,1,0,2,0,0,0,0,1,0,0,1,7,
	0,1,0,1,1,0,0,1,1,0,0,1,0,0,1,8,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,1,7,
	0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,2,
	1,1,1,1,1,0,2,0,1,0,0,1,0,0,0,7,
	0,1,1,1,0,0,2,0,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,2,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,3,
	1,1,1,1,1,0,0,1,1,1,0,1,0,0,0,7,
	1,0,0,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,1,5,
	0,0,1,1,0,0,0,1,0,1,0,1,0,1,0,1,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,1,1,0,1,2,1,1,1,0,1,0,0,0,5,
	1,1,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,8,
	1,1,0,1,1,0,0,0,1,1,0,1,0,1,0,4,
	0,0,0,1,1,0,2,0,0,0,0,1,0,1,1,8,
	0,1,1,1,0,0,2,0,0,1,0,1,0,0,1,8,
	1,1,1,1,1,1,2,1,1,0,0,1,1,0,1,8,
	0,1,1,1,0,0,2,0,1,1,0,1,0,1,0,5,
	0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,8,
	0,0,0,1,0,0,0,1,0,1,0,1,1,0,0,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,5,
	0,1,1,1,0,0,2,1,0,1,0,1,0,1,0,5,
	0,1,1,1,1,1,1,1,0,1,0,1,0,1,1,7,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,4,
	0,1,1,1,0,0,0,0,0,1,0,1,1,0,0,6,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,6,
	0,1,0,1,0,0,2,0,1,0,0,1,0,0,0,6,
	0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,3,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,1,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,1,0,0,1,1,1,0,1,0,1,0,3,
	0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,3,
	0,1,1,1,1,0,2,0,1,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,1,7,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,0,1,1,0,2,0,0,1,0,1,0,1,0,7,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,1,1,1,0,1,2,1,1,0,0,1,0,1,0,4,
	1,1,0,1,1,0,0,1,0,1,0,1,0,0,0,5,
	0,1,0,1,1,0,0,1,1,1,1,1,0,0,1,5,
	1,1,1,1,1,0,0,1,1,1,0,1,1,1,0,2,
	0,1,0,1,0,0,2,1,1,0,0,1,0,1,0,1,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,5,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,4,
	0,0,1,1,0,0,0,0,1,1,0,1,0,0,0,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,6,
	1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,4,
	1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,8,
	1,1,1,1,1,1,0,0,0,1,0,1,0,0,0,4,
	0,0,1,1,1,0,2,1,1,0,0,1,0,0,0,8,
	1,1,0,1,0,0,2,1,0,1,0,1,1,1,0,1,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,1,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,2,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,1,
	1,1,1,1,1,0,2,0,0,0,0,1,0,0,0,4,
	1,1,1,1,0,0,2,1,0,0,0,1,0,1,0,5,
	0,1,1,1,1,1,2,0,1,1,0,1,0,1,1,4,
	0,0,0,1,0,0,0,1,1,1,0,1,1,0,0,5,
	0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,8,
	0,1,1,1,0,1,2,0,1,1,0,1,0,0,0,7,
	0,1,1,1,0,0,2,0,1,0,0,1,0,1,0,3,
	0,1,1,1,1,0,2,0,0,0,0,1,0,1,0,1,
	0,1,0,1,1,0,0,1,1,1,0,1,1,1,0,2,
	1,1,1,1,1,0,2,1,0,1,0,1,0,1,1,2,
	0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,8,
	0,1,1,1,1,0,2,0,0,0,0,1,0,1,1,2,
	0,0,0,1,0,0,2,0,1,1,0,1,0,1,0,3,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,2,
	1,0,0,1,1,0,0,1,1,1,0,1,0,0,1,6,
	1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,3,
	0,0,0,1,1,0,0,1,1,0,0,1,0,0,1,7,
	0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,1,
	0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,1,
	0,1,1,1,0,0,2,0,1,1,0,1,0,0,0,6,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,0,5,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,7,
	1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,3,
	0,0,1,1,1,0,0,1,1,0,0,1,0,0,0,7,
	0,1,0,0,1,0,0,0,0,1,0,1,0,1,0,7,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,8,
	1,0,0,1,1,0,0,1,0,1,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,1,1,0,1,1,1,0,3,
	0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,2,
	0,0,0,1,0,0,0,0,1,1,0,1,0,1,1,6,
	1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,2,
	0,1,0,1,1,1,2,1,0,1,0,1,0,1,1,2,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,5,
	0,1,1,1,0,0,2,0,0,0,0,1,0,0,1,3,
	0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,3,
	0,0,1,1,1,0,2,0,1,0,0,1,0,1,1,6,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,2,
	0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,7,
	0,1,1,1,1,0,1,0,0,1,0,0,1,1,0,3,
	1,1,1,1,1,0,2,0,0,1,0,1,0,1,1,2,
	0,0,0,1,1,0,0,1,1,1,0,1,1,0,1,8,
	1,1,1,1,0,1,2,1,1,1,0,1,1,1,0,1,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,6,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,0,1,1,1,0,1,1,1,1,0,1,0,1,0,3,
	0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,6,
	0,1,1,1,0,0,2,0,1,1,0,1,0,1,0,2,
	0,1,1,1,1,0,2,1,1,1,0,1,1,0,0,2,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,2,
	0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,8,
	0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,7,
	0,1,0,1,0,0,2,1,0,1,0,1,0,0,1,1,
	0,1,0,1,0,1,2,1,0,1,0,1,0,1,0,4,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,2,1,1,1,0,1,0,0,0,8,
	0,0,1,1,0,0,2,1,1,1,0,1,0,0,0,1,
	0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,2,
	0,1,1,1,0,0,1,0,1,1,0,1,0,0,0,4,
	1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,3,
	0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,2,
	0,1,0,1,1,0,0,1,1,0,1,1,0,0,1,8,
	0,1,0,1,0,0,2,1,0,0,0,1,0,1,0,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,3,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,1,0,1,1,1,2,0,0,0,0,1,0,1,1,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,1,6,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,0,1,1,0,1,1,1,0,3,
	1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,4,
	0,1,1,1,0,0,2,1,1,1,0,1,0,1,0,1,
	1,1,1,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,5,
	1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,2,
	0,1,0,1,0,0,2,0,1,1,0,1,0,0,0,3,
	0,1,1,1,0,0,2,0,1,1,0,1,0,1,0,5,
	0,1,1,1,0,1,2,0,1,1,0,1,0,1,0,2,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,4,
	0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,
	0,1,1,1,1,0,2,0,0,1,0,0,0,0,0,6,
	0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,8,
	1,1,1,1,0,0,2,1,1,1,0,1,0,0,1,4,
	0,0,1,0,0,0,0,0,1,1,0,1,1,0,0,5,
	0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,7,
	0,1,0,1,1,0,2,1,1,1,0,1,0,1,0,1,
	0,0,1,1,0,0,0,0,0,1,0,0,1,0,1,3,
	1,1,0,1,1,0,0,0,0,0,1,1,0,1,1,8,
	1,1,1,1,0,0,0,0,1,1,0,1,0,1,0,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,1,1,3,
	0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,1,
	1,1,1,1,0,1,2,0,0,0,0,0,1,1,0,1,
	0,1,1,1,0,0,2,0,1,1,0,1,1,1,1,7,
	0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,4,
	0,1,1,1,1,1,2,1,0,0,1,1,0,1,1,5,
	1,1,1,1,0,1,2,0,0,1,0,1,1,1,1,1,
	0,1,0,1,1,0,0,1,0,0,0,1,0,0,1,5,
	0,1,1,1,1,0,0,0,1,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,1,0,1,0,0,2,0,0,0,0,1,0,1,0,1,
	0,1,0,1,0,0,0,1,1,0,0,1,0,0,0,3,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,2,
	0,1,1,1,1,0,0,1,1,1,0,1,1,0,1,6,
	0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,6,
	1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,1,0,0,1,0,1,1,1,0,0,0,8,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,7,
	0,1,1,1,1,0,0,0,1,1,0,1,0,0,0,3,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,1,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,2,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	1,1,0,1,0,1,0,0,1,0,0,1,1,0,0,3,
	0,1,1,1,0,0,2,1,0,1,0,1,0,0,0,5,
	0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,5,
	1,0,1,1,1,0,2,1,1,1,0,1,0,0,1,7,
	0,1,0,1,1,1,0,1,1,1,0,1,0,0,0,3,
	0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,5,
	0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,8,
	1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	1,1,1,1,1,0,0,1,1,1,0,1,0,0,0,3,
	0,0,1,1,1,0,0,1,1,1,0,1,1,1,0,3,
	0,0,0,1,1,0,2,0,0,1,0,0,1,0,0,3,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,0,0,1,1,0,2,1,1,1,1,1,0,0,0,8,
	0,1,0,1,0,0,2,0,1,1,0,1,0,0,0,3,
	0,1,1,1,0,1,2,0,0,0,0,1,0,1,1,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,0,1,0,0,2,1,0,0,0,1,0,1,1,4,
	0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,4,
	0,1,0,1,0,0,2,0,1,1,0,1,0,0,0,5,
	0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,0,2,
	0,0,0,1,1,0,0,0,1,0,0,1,1,1,0,2,
	0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,6,
	0,1,1,1,0,0,0,1,0,1,0,1,0,1,1,5,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,5,
	0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,7,
	0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,
	0,1,1,1,1,0,2,1,1,1,0,1,1,0,0,4,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,3,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,4,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,5,
	0,0,1,1,1,1,2,0,0,0,0,1,1,1,0,3,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,7,
	0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,7,
	0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,5,
	0,0,0,1,1,0,0,0,1,1,0,1,0,0,1,3,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,1,7,
	0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,4,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,3,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,0,4,
	0,1,1,1,1,0,0,0,0,1,0,1,0,1,1,8,
	1,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,4,
	0,1,1,1,0,0,0,0,1,0,0,1,0,1,0,1,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,0,0,1,1,1,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,1,0,2,1,1,1,0,1,0,0,0,6,
	0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,7,
	1,1,1,1,1,0,2,0,0,1,0,1,0,1,1,7,
	1,1,0,1,1,0,2,1,1,1,0,1,0,0,0,4,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,7,
	0,1,1,1,0,0,0,0,0,1,0,1,0,0,0,7,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	1,1,0,1,0,0,2,1,0,1,0,1,1,1,1,2,
	0,1,1,1,1,0,0,0,1,1,1,1,0,1,0,7,
	0,1,0,1,0,0,2,0,1,1,0,1,0,1,0,5,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,0,3,
	0,1,1,1,0,0,2,1,0,0,0,1,1,1,0,2,
	0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,2,
	0,1,0,1,0,0,0,1,0,1,0,0,1,1,0,4,
	1,1,1,1,1,0,2,0,1,1,0,1,0,1,1,6,
	1,1,0,1,1,0,2,1,0,0,0,1,0,0,1,8,
	0,1,1,1,0,0,2,0,0,0,0,1,0,0,0,7,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,3,
	0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,8,
	0,1,1,1,1,0,0,0,1,1,1,1,0,1,0,6,
	0,1,1,1,0,0,2,0,1,1,0,1,0,0,0,3,
	0,1,1,1,0,0,0,1,0,1,0,1,0,1,0,6,
	0,0,1,1,0,0,2,1,0,1,0,1,0,0,1,8,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,7,
	0,0,1,1,1,0,0,0,1,0,0,1,0,0,1,4,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,0,5,
	0,0,0,1,1,0,2,0,1,1,0,1,0,1,0,5,
	0,1,1,1,1,0,2,0,0,0,0,1,0,1,1,6,
	0,0,0,1,1,0,0,1,0,1,0,1,1,1,0,6,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,5,
	1,1,1,1,0,0,2,1,1,1,0,1,0,0,1,8,
	0,1,0,1,1,0,2,1,0,0,0,1,0,0,1,6,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,0,0,1,0,0,2,1,1,1,0,0,0,0,0,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,0,5,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,4,
	0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,1,1,0,0,2,1,1,1,0,0,0,0,0,4,
	1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,1,
	0,1,1,1,1,0,2,1,0,1,0,1,0,0,1,7,
	1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,6,
	1,1,1,1,1,0,2,0,1,1,0,1,0,0,1,7,
	1,1,1,1,1,0,2,1,1,1,0,1,0,1,0,8,
	0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,4,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,8,
	1,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,2,
	0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,7,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	1,1,1,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,0,1,1,0,0,1,1,1,0,1,0,1,1,8,
	0,1,0,1,1,0,0,1,1,0,0,1,0,0,0,2,
	0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,1,
	1,1,1,1,0,0,2,1,1,0,0,1,1,1,0,1,
	0,1,1,1,0,0,0,0,0,1,0,1,0,1,0,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,2,0,0,0,0,1,1,1,0,1,
	1,1,0,1,1,1,2,1,0,0,0,1,0,1,0,1,
	0,1,1,1,0,0,2,1,1,1,0,1,0,1,0,3,
	0,1,0,1,0,0,2,1,1,0,0,1,0,0,0,7,
	0,1,1,1,1,0,2,0,0,0,0,1,0,1,1,2,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,1,1,1,0,0,0,0,0,1,0,1,0,1,0,3,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,0,0,1,1,0,2,0,1,1,0,1,0,0,1,5,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,2,1,1,1,0,1,0,1,0,4,
	0,1,1,1,1,0,2,1,0,0,0,1,0,1,0,2,
	0,0,0,1,0,0,2,1,1,1,0,1,0,0,0,8,
	0,0,1,1,1,0,0,1,0,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,4,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,2,
	0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,6,
	1,0,0,1,1,0,2,0,0,0,0,1,0,1,1,3,
	0,0,1,1,1,0,2,1,1,1,0,1,0,1,0,6,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,2,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,8,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,4,
	0,0,1,1,0,0,2,1,0,1,0,1,1,0,0,4,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,6,
	0,1,0,1,1,0,2,1,0,1,0,1,0,0,1,3,
	0,1,1,1,0,0,0,0,0,1,0,1,0,1,0,2,
	0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,6,
	0,0,0,1,0,0,2,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,0,1,1,0,0,1,1,0,0,2,
	0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,2,
	0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,6,
	0,0,0,1,1,0,0,1,1,0,0,1,0,1,1,8,
	1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,2,
	0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,5,
	0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,2,
	0,1,1,1,1,0,1,1,0,1,0,1,0,1,0,5,
	0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,3,
	1,1,0,1,0,1,2,0,1,1,0,1,0,1,0,6,
	1,1,1,1,0,0,0,1,1,1,0,1,0,1,0,4,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	1,1,1,1,0,1,2,1,1,1,0,1,0,1,0,6,
	0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,8,
	0,1,1,1,1,0,2,0,0,0,0,1,0,1,1,3,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,6,
	0,1,0,1,0,0,2,1,0,1,0,1,0,0,0,5,
	1,1,1,1,0,1,2,1,1,0,0,1,1,1,0,3,
	0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,4,
	0,1,1,1,0,0,1,0,0,0,0,1,0,0,0,4,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,7,
	0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,3,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,1,0,1,1,0,0,0,0,0,1,1,0,0,1,6,
	0,1,0,1,0,0,0,0,0,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,6,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,0,1,0,1,0,1,0,5,
	0,1,0,1,0,0,2,0,1,0,0,1,0,1,0,1,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,7,
	0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,5,
	0,1,1,1,0,0,2,1,1,1,0,1,0,0,0,6,
	0,1,1,1,1,0,0,0,0,0,0,1,0,1,1,3,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,3,
	0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,7,
	0,0,1,1,0,0,2,1,0,0,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,6,
	0,1,1,1,0,0,2,1,1,1,0,1,0,0,0,3,
	1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,4,
	0,1,0,1,0,0,2,1,0,1,0,1,0,0,0,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,7,
	1,1,1,1,1,0,2,0,0,1,1,1,0,0,1,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,3,
	0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,0,1,1,0,1,1,1,0,1,
	0,1,1,1,1,0,1,0,1,1,0,1,0,0,0,7,
	0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,6,
	1,1,1,1,0,0,0,0,1,0,0,1,0,1,0,2,
	0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,0,0,0,1,0,1,0,0,0,3,
	0,1,1,1,1,0,2,1,0,0,0,1,0,1,0,1,
	0,1,0,1,1,0,2,1,1,1,0,1,0,0,1,8,
	1,0,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,7,
	0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,
	0,1,1,1,0,0,2,0,1,1,0,1,0,0,0,5,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,0,3,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,1,0,1,0,0,2,1,1,0,0,0,1,0,0,1,
	0,1,1,1,1,1,2,0,1,1,0,1,1,1,0,3,
	0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,0,1,1,1,0,0,1,0,0,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,4,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,3,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,5,
	0,0,1,1,0,0,0,0,1,1,0,1,0,1,0,4,
	0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,7,
	1,0,1,1,1,0,0,0,1,1,0,1,0,1,0,6,
	0,1,1,1,0,0,2,1,0,0,0,1,0,0,0,2,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,5,
	0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,3,
	0,0,1,1,1,0,2,1,0,1,0,1,0,0,1,6,
	0,1,0,1,0,0,2,1,1,1,0,1,1,1,1,3,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,0,7,
	0,1,1,1,1,0,0,1,1,1,1,1,1,0,0,7,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,0,1,1,0,0,2,1,1,0,0,1,0,0,0,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,6,
	0,0,0,1,1,0,2,0,1,1,0,1,1,1,0,3,
	0,1,0,1,1,0,0,0,0,1,0,1,0,0,1,8,
	1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,5,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,7,
	0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,2,
	0,0,0,1,1,0,0,0,0,1,0,1,0,1,1,1,
	0,1,0,1,1,0,0,1,1,1,0,0,1,0,0,7,
	0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,6,
	0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,1,0,0,1,0,1,0,1,1,1,1,3,
	0,1,0,1,0,1,0,0,1,1,0,1,0,0,1,6,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,5,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,3,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,7,
	0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,4,
	1,1,0,1,1,0,2,1,0,1,0,1,0,0,1,4,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,1,1,1,0,0,1,0,1,0,1,0,0,0,4,
	0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,6,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,7,
	0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,3,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,7,
	0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,6,
	0,1,0,1,0,0,2,0,1,1,0,1,0,1,0,6,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,2,1,0,0,0,0,0,1,0,2,
	0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,6,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	1,1,1,1,1,0,0,1,0,1,0,1,0,1,1,3,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,6,
	0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,3,
	0,1,1,1,1,1,0,1,0,1,0,1,0,0,1,7,
	0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,7,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,3,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,6,
	0,1,1,1,0,0,2,0,1,0,0,1,0,1,0,3,
	0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,1,1,1,0,0,0,0,1,0,1,1,1,0,3,
	0,0,1,1,0,0,0,1,1,0,0,1,0,0,1,4,
	0,1,1,1,1,0,2,1,0,1,0,1,0,0,0,2,
	0,1,0,1,1,1,0,1,1,1,1,1,0,0,1,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	1,1,1,1,1,0,0,1,0,1,0,1,0,0,1,4,
	0,1,0,1,0,0,2,0,1,1,0,1,0,1,1,8,
	1,1,1,1,1,0,2,1,1,1,0,1,0,0,0,5,
	0,0,1,1,0,0,0,1,1,1,0,1,0,1,1,1,
	0,1,1,1,1,0,2,0,1,1,0,1,0,1,1,8,
	0,1,1,1,0,0,2,0,1,1,0,1,0,0,1,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,1,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,1,1,1,1,0,2,1,0,1,0,1,0,1,1,6,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,2,
	0,0,0,1,0,0,2,1,0,1,0,1,0,0,0,8,
	0,0,1,1,1,0,0,1,0,1,0,1,0,0,0,5,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,2,0,1,1,0,1,0,1,1,6,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,2,
	0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,0,1,1,0,1,0,1,1,1,0,1,0,0,0,8,
	0,0,1,1,1,0,0,1,0,1,0,1,0,0,0,6,
	0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,3,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	1,0,1,1,0,0,0,0,1,1,0,1,0,1,1,6,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,7,
	0,0,1,1,1,0,2,1,0,1,0,1,0,0,1,6,
	0,1,0,1,1,0,2,0,1,1,0,1,0,1,1,5,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,1,3,
	0,1,1,1,0,0,2,1,1,1,0,1,0,0,0,2,
	0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,2,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,8,
	1,1,0,1,0,0,2,0,0,1,0,1,0,1,1,5,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,0,1,1,0,1,0,1,1,1,0,0,0,0,1,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,1,0,1,1,0,0,0,0,0,0,1,0,1,1,5,
	0,1,1,1,0,0,2,0,1,1,0,1,0,1,0,7,
	0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,8,
	1,0,1,1,0,1,2,1,1,1,0,1,1,1,0,3,
	0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,5,
	0,0,0,1,0,0,2,0,1,1,0,1,1,1,0,7,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,5,
	0,1,1,1,0,0,2,1,0,1,0,1,0,0,0,8,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,7,
	0,1,1,1,1,1,0,0,1,1,0,1,0,1,0,3,
	1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,6,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,5,
	0,1,1,1,1,0,2,1,0,1,0,1,0,0,1,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,1,
	0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,6,
	0,0,0,1,1,0,0,1,1,1,0,0,1,0,1,4,
	0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,8,
	1,1,1,1,0,0,2,1,0,0,0,1,0,0,0,7,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,7,
	0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,8,
	0,0,1,1,1,0,0,0,1,1,0,1,0,1,1,3,
	0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,1,
	0,0,1,1,0,0,0,0,1,0,0,1,0,1,0,3,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,7,
	0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,5,
	0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,3,
	0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,2,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,1,1,1,0,0,2,1,1,1,0,1,0,0,0,3,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,3,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,5,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,2,
	0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,3,
	0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,8,
	0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,4,
	0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,5,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,1,1,1,0,0,0,1,0,0,0,1,0,0,1,8,
	0,1,0,1,0,0,2,1,0,1,0,1,0,0,1,7,
	0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,6,
	1,1,0,1,1,0,2,0,0,0,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,8,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,3,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,5,
	1,0,1,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,3,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,7,
	0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,8,
	1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,5,
	0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,8,
	0,0,1,1,1,0,0,0,0,1,0,1,0,0,1,5,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,5,
	0,0,1,1,1,0,0,0,0,1,0,1,0,1,1,6,
	1,1,1,1,0,0,0,1,0,1,0,0,1,1,1,5,
	0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,6,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,7,
	0,1,1,1,1,0,2,0,1,0,0,1,0,1,0,4,
	0,0,0,1,0,1,0,0,1,0,0,1,0,0,1,8,
	1,1,1,1,0,0,0,1,0,1,0,1,1,0,0,5,
	1,1,1,1,1,1,2,0,1,1,0,1,0,1,1,7,
	1,1,1,1,0,1,2,0,1,1,0,1,0,1,0,5,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,1,0,1,1,0,0,0,0,1,1,1,1,0,0,7,
	0,1,1,1,1,0,2,1,1,1,0,1,0,0,0,3,
	0,1,1,1,1,1,2,0,1,1,0,1,0,0,0,5,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,1,1,0,0,2,1,1,1,0,1,0,1,0,5,
	1,0,0,1,0,1,2,1,1,1,0,1,0,1,0,8,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	1,1,0,1,0,0,0,1,1,1,0,1,0,1,0,2,
	0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,8,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,0,1,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,4,
	0,0,0,1,1,0,2,0,0,0,0,1,0,1,0,1,
	0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,3,
	1,1,1,1,0,0,0,0,0,1,0,1,0,1,1,2,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,
	0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,2,
	0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,8,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	1,0,1,1,0,0,0,0,0,0,0,1,0,0,1,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,6,
	0,1,1,1,1,0,0,1,1,1,0,1,1,0,0,1,
	0,1,1,1,1,0,0,1,1,0,0,1,0,0,0,4,
	0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,8,
	0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,7,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,7,
	1,1,0,1,1,0,0,1,1,1,1,1,0,0,0,8,
	0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,5,
	1,0,0,1,0,0,2,0,1,0,0,1,0,0,0,1,
	0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,5,
	1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,8,
	0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,4,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,1,7,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,6,
	0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,8,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,6,
	1,1,0,1,1,0,0,1,1,1,0,1,0,0,1,7,
	0,1,1,0,1,0,0,0,0,0,0,1,0,0,1,8,
	0,0,0,1,0,0,2,1,1,1,0,1,0,1,1,5,
	0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,2,
	0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,5,
	1,1,1,1,1,0,0,0,1,0,0,1,1,1,1,6,
	0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,7,
	0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,3,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,1,0,5,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,1,6,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,8,
	0,0,0,1,1,0,0,1,0,0,0,1,1,0,0,3,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,1,5,
	0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,
	0,1,1,1,0,0,2,1,1,1,0,1,0,1,0,2,
	0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,
	1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,7,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,6,
	0,1,0,1,1,0,2,1,1,1,0,1,0,0,0,8,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,2,
	0,0,1,1,1,0,0,1,1,1,0,1,0,1,1,7,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,0,2,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,8,
	1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,3,
	1,1,1,1,0,0,0,0,1,1,0,1,0,1,0,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,1,0,1,0,0,1,1,1,1,0,1,0,0,0,4,
	0,0,0,1,1,0,0,1,0,1,0,1,0,0,0,8,
	1,1,1,1,1,1,2,0,0,1,0,1,0,1,0,6,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	1,1,1,1,0,0,0,0,1,1,0,1,0,0,1,6,
	0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,6,
	0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,2,
	0,1,1,1,1,0,0,0,0,1,0,1,0,1,1,1,
	0,1,0,1,1,0,2,1,1,1,0,1,0,1,1,3,
	0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,4,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,7,
	0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,4,
	1,1,1,1,0,1,2,0,0,1,0,1,0,1,0,3,
	0,1,1,1,0,0,0,1,1,1,0,1,0,1,0,2,
	1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,2,
	0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,8,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,5,
	0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,4,
	0,1,0,1,1,0,2,1,0,0,0,1,0,1,0,5,
	1,0,1,1,0,0,2,0,0,0,0,1,0,0,1,7,
	0,1,1,1,1,0,2,1,0,1,0,1,0,0,0,5,
	0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,5,
	1,1,1,1,1,0,2,0,1,1,0,1,0,0,1,6,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,4,
	0,1,0,1,1,0,2,1,1,1,0,1,0,0,0,4,
	1,1,1,1,1,0,2,0,1,1,0,1,0,1,1,6,
	0,1,0,1,0,0,2,0,1,1,0,1,0,1,0,6,
	1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,8,
	0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,4,
	0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,8,
	0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,7,
	1,1,1,1,0,0,2,0,0,0,0,1,0,0,1,7,
	0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,6,
	0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,8,
	0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,5,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,8,
	0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,8,
	0,1,1,1,1,0,2,1,1,1,0,1,0,0,0,3,
	0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,7,
	0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,4,
	1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,4,
	0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,7,
	1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,4,
	0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,5,
	1,1,1,1,0,0,0,0,1,0,0,1,0,1,0,1,
	1,0,1,1,0,0,0,1,0,1,0,1,1,1,0,5,
	0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,1,
	0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,8,
	0,1,0,1,1,0,2,1,0,0,0,1,0,0,0,1,
	0,1,1,1,1,0,0,1,0,1,1,1,0,0,0,8,
	0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,4,
	0,1,0,1,1,0,0,0,1,0,0,1,1,1,1,3,
	1,1,1,1,0,0,0,0,0,0,0,1,0,1,0,1,
	0,1,0,1,0,0,2,0,1,1,0,1,1,1,0,2,
	0,1,0,1,1,0,0,1,1,1,0,1,0,0,0,7,
	0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,3,
	1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,4,
	1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,7,
	0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,7,
	0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,3,
	0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,2,
	1,1,0,1,1,0,2,1,1,0,0,1,0,0,0,4,
	0,0,0,1,0,0,0,1,0,1,0,1,1,1,0,3,
	0,1,1,1,0,0,0,1,1,0,0,1,0,0,0,1,
	0,1,1,1,0,0,0,1,0,1,0,1,0,1,0,6,
	0,1,0,1,1,1,0,1,0,1,0,1,0,0,1,1	
	};
  	std::vector<XT> features;
    std::vector<YT> labels;	
	
	int n = 0;
    for (size_t i=0; i<hd_data.size(); i += 16) {
    	for (int j=1; j<15; ++j) features.push_back(hd_data[i+j]);
        labels.push_back(static_cast<YT>(hd_data[i]));
        n++;
    }	
	X.resize(n, 15);
	X.set(features);
	y.resize(n); 
	y.set(labels);
}


};     // namespace umml

#endif // UMML_BREASTCANCER_INCLUDED
