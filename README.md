# Boop-AI

A web-based artificial intelligence opponent for the board game **Boop**, developed as part of a senior Computer Science project.

## 📌 Project Overview

This project uses the [SIMPLE](https://github.com/davidADSP/SIMPLE) reinforcement learning library to train an AI agent that plays Boop, a strategic 2-player game focused on pattern recognition and positioning. The AI learns through self-play within a custom-coded environment that implements the official game rules. Users can interact with the AI through a lightweight browser interface.

Key features:
- A Python-based game engine implementing Boop’s rules
- AI training via reinforcement learning using SIMPLE
- A web interface built with **Brython** and **FastAPI**
- Play options: human vs. AI, human vs. human, and AI vs. AI

## ⚙️ Technologies Used

- **Python 3**
- **FastAPI** – backend server
- **Brython** – Python in the browser
- **SIMPLE** – reinforcement learning environment
- **HTML/CSS** – styling and layout

## 🚀 Setup Instructions

1. **Clone the repository**
2. **Create a Virtual Environment**
   python 3 -m venv venv
   source venv/bin/activate
4. **Install Dependencies**
   bash dependencies.txt
   * NOTE: Make sure the Python interpreter is version 3.11
5. **Train the Model**
   python3 boop_selfplay.py
6. **Enjoy Playing the Game**
   python3 api_server.py

## 👥 Project Collaboration

This project was originally developed as part of a team during our senior capstone course. This repository is a personal fork reflecting contributions, including:
- Designing and implementing the browser-based UI using Brython
- Creating FastAPI routes to manage game state and interactions
- Collaborating on frontend/backend integration, gameplay logic, and AI integration
