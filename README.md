# Bio-Ionic-Memristor

This project focuses on designing and analyzing a bio-ionic memristor using natural materials such as honey to emulate synaptic behavior for neuromorphic computing systems.
Neuromorphic computing aims to mimic the human brain by integrating memory and computation into a single element, enabling efficient and adaptive learning systems.

Unlike traditional hardware, this project explores low cost fabrication, eco-friendly material and energy efficient AI in hardware.

# Machine Learning Component

A Linear Regression Model is used to predict conductance evolution.
Model Concept: G(t+1) = G(t) + \Delta G
Where:
	•	Inputs include voltage pulses, time, and previous conductance states
	•	Output predicts future conductance

Purpose:
	•	Predict device behavior
	•	Enable AI-assisted hardware optimization

# Methodology

<img width="864" height="407" alt="Screenshot 2026-04-19 at 3 17 51 PM" src="https://github.com/user-attachments/assets/fdcdc53f-f986-4aff-92a1-d505393a617e" />

A minimal hardware–software platform to characterize a bio-ionic memristor using an ESP32, INA219, and Python. The system applies voltage pulses/sweeps, measures current response, and logs data for analysis.

# Key outcomes:
I–V hysteresis and switching (Vset, Vreset)
ON/OFF ratio
Retention and endurance
Synaptic behavior (STP, LTP, PPF)
ESP32 generates stimuli, INA219 senses current, and Python handles control, logging, and analysis.

# Observations
40% honey concentration → Best performance
30% honey concentration → unstable (high leakage)
90% honey concentraction → strong switching but poor durability
device shows: clear resistive switching, synaptic-like behavior and conductance saturation after ~15–20 pulses

