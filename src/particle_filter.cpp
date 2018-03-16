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
#include <functional>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  is_initialized = true;
  num_particles = 100;
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    weights.push_back(1.0);
    particles.push_back(p);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  
  for(int i = 0; i < num_particles; i++) {
    Particle *p = &particles[i];
    double xf = 0;
    double yf = 0;
    double thetaf = 0;
    if(fabs(yaw_rate) > 0.0001) {
      xf = p->x + (velocity/yaw_rate)*(sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
      yf = p->y + (velocity/yaw_rate)*(cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
      thetaf = p->theta + yaw_rate * delta_t;
    } else {
      xf = p->x + velocity * delta_t * cos(p->theta);
      yf = p->y + velocity * delta_t * sin(p->theta);
      thetaf = p->theta;
    }
    normal_distribution<double> dist_x(xf, std_pos[0]);
    normal_distribution<double> dist_y(yf, std_pos[1]);
    normal_distribution<double> dist_theta(thetaf, std_pos[2]);

    p->x = dist_x(gen);
    p->y = dist_y(gen); 
    p->theta = dist_theta(gen);
  }
}

double square(double x) {
  return x * x;
}

void transformObservation(Particle *p,
                          const std::vector<LandmarkObs> &observations,
                          std::vector<LandmarkObs> &transformed_observations) {
  double x_p = p->x;
  double y_p = p->y;
  for (int j = 0; j < observations.size(); j++) {
    LandmarkObs o = observations[j];
    double x_c = o.x;
    double y_c = o.y;

    //convert observation(from car) coordinates to map coordinates
    //c2m -> car to map
    double x_c2m = x_p + (cos(p->theta) * x_c) - (sin(p->theta) * y_c);
    double y_c2m = y_p + (sin(p->theta) * x_c) + (cos(p->theta) * y_c);

    LandmarkObs t;
    t.id = o.id;
    t.x = x_c2m;
    t.y = y_c2m;
    transformed_observations.push_back(t);
  }
}

void getLandmarks(Particle *p,
                  std::vector<LandmarkObs> &list_of_landmarks, 
                  double sensor_range,
                  const Map &map_landmarks) {
  double x_p = p->x;
  double y_p = p->y;
  for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {
    Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
    double distance = dist(x_p, y_p, landmark.x_f, landmark.y_f);
    if(distance < sensor_range) {
      LandmarkObs t;
      t.id = landmark.id_i;
      t.x = landmark.x_f;
      t.y = landmark.y_f;
      list_of_landmarks.push_back(t);
    }
  }
}

void associate(std::vector<LandmarkObs> transformed_observations,
               std::vector<LandmarkObs> list_of_landmarks,
               std::vector<std::tuple<LandmarkObs, LandmarkObs>> &obs_landmark_assoc) {
  for(int i = 0; i < transformed_observations.size(); i++) {
    LandmarkObs obsL = transformed_observations[i];
    double minDistance = 10000000.0;
    LandmarkObs least;
    for(int j = 0; j < list_of_landmarks.size(); j++) {
      LandmarkObs predL = list_of_landmarks[j];
      double distance = dist(obsL.x, obsL.y, predL.x, predL.y);
      if(distance < minDistance) {
        least = predL;
        minDistance = distance;
      }
    }
    obs_landmark_assoc.push_back(std::make_tuple(obsL, least));
  }
}

void calculate_weight(LandmarkObs obsL,
                      LandmarkObs landmark,
                      double std_landmark[],
                      double &weight) {
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double x_obs = obsL.x;
  double y_obs = obsL.y;
  double mu_x = landmark.x;
  double mu_y = landmark.y;

  double gauss_norm = (1.0/(2.0 * M_PI * sig_x * sig_y));
  double exponent = square(x_obs - mu_x)/(2 * square(sig_x)) + square(y_obs - mu_y)/(2 * square(sig_y));
  weight = gauss_norm * exp(-exponent);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  
  //Clear weights
  weights.clear();
  for (int i = 0; i < particles.size(); i++) {
    Particle *p = &particles[i];

    //Clear vectors and reset weight to 1.0
    p->sense_x.clear();
    p->sense_y.clear();
    p->associations.clear();
    p->weight = 1.0;
    
    //Transform observation from car coordinates to map coordinates
    std::vector<LandmarkObs> transformed_observations;
    transformObservation(p, observations, transformed_observations);

    //List the landmarks that is within sensor range
    std::vector<LandmarkObs> list_of_landmarks;
    getLandmarks(p, list_of_landmarks, sensor_range, map_landmarks);

    //Associate transformed observations with landmark
    std::vector<std::tuple<LandmarkObs, LandmarkObs>> obs_landmark_assoc;
    associate(transformed_observations, list_of_landmarks, obs_landmark_assoc);

    //Calculate the weight and final weight
    for(int i = 0; i < obs_landmark_assoc.size(); i++) {
      auto assoc = obs_landmark_assoc[i];
      LandmarkObs obsL = std::get<0>(assoc);
      LandmarkObs landmark = std::get<1>(assoc);
      //Push the landmark id and its transformed observation x,y coordinates 
      p->associations.push_back(landmark.id);
      p->sense_x.push_back(obsL.x);
      p->sense_y.push_back(obsL.y);

      double weight;
      calculate_weight(obsL, landmark, std_landmark, weight);

      p->weight *= weight;
    }
    weights.push_back(p->weight);
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;
  std::discrete_distribution<> distribution(weights.begin(), weights.end());

  std::vector<Particle> resampled;
  for(int i = 0; i < num_particles; i++) {
    resampled.push_back(particles[distribution(gen)]);
  }

  particles = resampled;
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
