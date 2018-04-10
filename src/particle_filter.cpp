/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if(!is_initialized){

		num_particles = 30;
		default_random_engine gen;
		weights.resize(num_particles);

		//create a nomal distribution for x, y and theta.
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		//Initialising the particles
		for (int i = 0; i < num_particles; i++) {
			Particle particle;
			particle.id = i;
			particle.x = dist_x(gen);
			particle.y = dist_y(gen);
			particle.theta = dist_theta(gen);
			particle.weight = 1.0;
			particles.push_back(particle);
		}
		is_initialized = true;
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	//Creating the normal distribution for noise
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	for(int i = 0; i<num_particles; ++i){
		if(fabs(yaw_rate) > 0.001){
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t))- sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta)- cos(particles[i].theta + (yaw_rate*delta_t)));
			particles[i].theta += yaw_rate*delta_t;
		}
		else{
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);

		}

		//Add Noise
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	int id;
	for(int i = 0; i< observations.size(); i++){
		double max_val = numeric_limits<double>::max();
		for(int j = 0; j < predicted.size(); j++){
			double distance = dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y); 
			if(distance<max_val){
				id = j;
				max_val = distance;
			}
		}
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i = 0; i < num_particles; i++){
		std::vector<LandmarkObs> transformedObs;
		std::vector<LandmarkObs> predlandmarkObs;
		//transform the vehicle coordinates to the map coordinates
		for(int j = 0; j < observations.size(); j++){
			LandmarkObs landmark;
			landmark.id = observations[j].id;
			landmark.x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y);
			landmark.y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y);
			transformedObs.push_back(landmark);
		}
		//finding the landmarks in the sensor range for each particle
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
			double distance = (map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particles[i].x, particles[i].y);
			if(distance < sensor_range){
				LandmarkObs predLandmark;
				predLandmark.id = map_landmarks.landmark_list[j].id_i;
				predLandmark.x = map_landmarks.landmark_list[j].x_f;
				predLandmark.y = map_landmarks.landmark_list[j].y_f;
				predlandmarkObs.push_back(predLandmark);
			}
		}
		dataAssociation(predlandmarkObs,transformedObs);

		//set the associations
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		for(int j = 0; j < transformedObs.size(); j++){
			sense_x.push_back(transformedObs[j].x);
			sense_y.push_back(transformedObs[j].y);
			associations.push_back(predlandmarkObs[transformedObs[j].id].id);
		}

		SetAssociations(particles[i],associations,sense_x,sense_y);
		double std_xland = std_landmark[0];
		double std_yland = std_landmark[1];
		double prob = 1.0;
		
		//Particle Weights
		for(int j = 0; j < observations.size(); j++){
			double exponent_x = ((transformedObs[j].x - predlandmarkObs[transformedObs[j].id].x)*
			(transformedObs[j].x - predlandmarkObs[transformedObs[j].id].x))/(2*std_xland*std_xland);
			double exponent_y = ((transformedObs[j].y - predlandmarkObs[transformedObs[j].id].y)*
			(transformedObs[j].y - predlandmarkObs[transformedObs[j].id].y))/(2*std_yland*std_yland);
			double exponent = exponent_x+exponent_y;
			double normalization_term = 1/(2*M_PI*std_xland*std_yland);
			prob *= exp(-(exponent))*normalization_term;
			transformedObs.clear();
			predlandmarkObs.clear();
		}
		particles[i].weight = prob;
		weights[i] = prob;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std:vector<Particle> resampled_particles;
	default_random_engine gen;
	double min_val = numeric_limits<double>::min();
	for(int i =0; i< num_particles; i++){
		if(particles[i].weight > min_val){
			min_val = particles[i].weight; 
		}
	}

	//creating uniform distributions for particles and weights
	uniform_int_distribution<int> dist_particles(0, num_particles-1);
	uniform_real_distribution<double> dist_weights(0, min_val);

	int index = dist_particles(gen);
	double beta = 0.0;
	for(int i = 0; i < num_particles; i++){
		beta += dist_weights(gen) * 2 * min_val;
		while(particles[index].weight<beta){
			beta -= particles[index].weight;
			index = (index+1)%num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
