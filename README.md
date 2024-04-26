
# RecoveringSleepingBandit

# Duolingo Multi-Armed Bandit Optimization Algorithm

This repository replicates the multi-armed bandit optimization algorithm implemented by Duolingo researchers Kevin P. Yancey and Burr Settles in their paper "A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications". The algorithm uses historical choices of notifications meant to engage the user to use Duolingo to inform the best notification to send to the user at any time. Success is measured by the user completing their lesson within 2 hours of receiving the notification. The algorithm introduces novel concepts like applying a recency penalty to notifications sent recently due to the novelty effect (users becoming desensitized to the same frequent notification). The algorithm implemented here resulted in a 1.52% increase in average reward compared to the baseline policy of choosing a notification at random. This project was built in Python, utilizes a `parquet_reader` to convert Parquet data into a Pandas DataFrame, implements a custom deque data structure to control historical rounds, and implements the proposed algorithm on open-sourced data from Duolingo.

The main algorithm can be found in the 'algorithm_bandit.py' file, along with a language + arm implementation in 'algorithm_lang_arms.py'

## Table of Contents

1. [Introduction](#introduction)
2. [Technologies and Skills Used](#technologies-and-skills-used)
3. [Purpose](#purpose)
4. [Results](#results)
5. [Contributing](#contributing)

## Introduction

This repository contains an implementation of the multi-armed bandit optimization algorithm described in the paper by Kevin P. Yancey and Burr Settles. The algorithm is designed to optimize recurring notifications in educational platforms like Duolingo, with the goal of increasing user engagement and completion rates. It controls for many variables previously unaddressed in real-world bandit optimization problems.

## Technologies and Skills Used

- Python
- Custom deque data structure
- Data analysis and manipulation
- Machine learning algorithms
- Open-source data handling

## Purpose

The purpose of this project is to replicate and implement the multi-armed bandit optimization algorithm proposed by Yancey and Settles. By doing so, we aim to demonstrate its effectiveness in improving user engagement metrics, specifically in the context of educational platforms such as Duolingo.

## Results

The algorithm implemented in this specific project resulted in a 1.52% increase in average reward compared to the baseline policy of choosing a notification at random, which is comparable to the results achieved by Duolingo researchers. This improvement signifies the algorithm's ability to learn and adapt over time, leading to better outcomes in terms of user behavior and engagement.


## Contributing

Contributions to this project are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.
>>>>>>> 87eb123dcfc6ffd9ec1e8ada7e2e475ad12e6a11
